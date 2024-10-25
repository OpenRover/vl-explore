# ==============================================================================
# Author: Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
# Multi-thread daemonized Unix socket server and client (line based protocol).
# ==============================================================================
import errno, time, atexit
from typing import Generator, Iterable, TypeVar, Generic
from socket import socket, AF_UNIX, SOCK_STREAM
from pathlib import Path
from io import BufferedIOBase

from .transport import Transport
from .logger import Logger
from .exception import Expect, no_throw
from . import JSON

logger = Logger(__file__)

P = TypeVar("P")

class Protocol(Generic[P]):
    sep: str = ","
    end: str = "\n"

    @classmethod
    def encode(cls, item: P) -> Generator[str, None, None]:
        raise NotImplementedError

    @classmethod
    def decode(cls, line: str) -> Generator[P, None, None]:
        raise NotImplementedError


class DefaultProtocol(Protocol[str]):  # Python print-like dummy protocol
    sep: str = ","
    end: str = ""

    @classmethod
    def encode(cls, item):
        yield item

    @classmethod
    def decode(cls, line):
        yield line


class JsonProtocol(Protocol[P]):
    sep: str = ","
    end: str = "\n"

    @classmethod
    def to_items(cls, item: P) -> Iterable:
        assert isinstance(item, Iterable)
        yield from item

    @classmethod
    def from_items(cls, items: list) -> Generator[P, None, None]:
        yield items

    @classmethod
    def encode(cls, item: P):
        assert isinstance(item, Iterable), f"Item <{item}> not iterable"
        for item in cls.to_items(item):
            yield JSON.stringify(item)

    @classmethod
    def decode(cls, line: str):
        try:
            line = f"[{line}]"
            result: list = JSON.parse(line)
            yield from cls.from_items(result)
        except Exception as e:
            logger.debug(f"[Protocol {cls.__name__}] Failed to parse line: {e}")
            L = 128
            if len(line) > L:
                line = line[:L] + f" ({len(line) - L} chars omitted) ..."
            logger.debug(f"Line content: {line}")


class Readlines:
    def __init__(self, block_size: int = 4096):
        self.buffer = ""
        self.block_size = block_size

    def __call__(self, s: socket | BufferedIOBase):
        try:
            if isinstance(s, socket):
                byte = s.recv(self.block_size)
            else:
                assert hasattr(s, "read"), f"{s} has no read method"
                byte = s.read(self.block_size)
                if byte is None:
                    return
            if not len(byte):
                raise OSError(errno.ECONNABORTED)
            self.buffer += byte.decode(encoding="utf-8")
            if "\n" in self.buffer:
                lines = self.buffer.split("\n")
                self.buffer = lines.pop()
                for line in lines:
                    yield line + "\n"
        except TimeoutError:
            return
        except OSError as e:
            reason = e.args[0]
            if reason in (errno.EAGAIN, errno.EWOULDBLOCK, errno.ETIMEDOUT):
                return
            if reason in (errno.ECONNRESET, errno.ECONNABORTED):
                s.close()
                return
            else:
                raise RuntimeError(f"Readlines error: {type(reason)} {reason}")


class SocketTransport(Transport[P, P]):
    logger: Logger = logger
    RX: bool = True
    TX: bool = True

    def __init__(self, path: Path, protocol: type[Protocol[P]] = DefaultProtocol, **kw):
        super().__init__(**kw)
        assert issubclass(protocol, Protocol)
        self.protocol = protocol
        assert isinstance(path, Path)
        self.path = path
        self.sockets: set[tuple[socket, Readlines]] = set()
        atexit.register(self.close)
    
    def __str__(self):
        return f"{self.__class__.__name__}(UNIX:{self.path.name})"

    def __repr__(self):
        return str(self)

    def spin(self):
        # Remove disconnected clients
        inactive: set[socket] = set()
        for item in self.sockets:
            c, rl = item
            if c.fileno() == -1:
                inactive.add(item)
        self.sockets.difference_update(inactive)
        del inactive
        # Only read from clients if RX is enabled
        if not self.RX:
            return
        # Dump all incoming messages
        for s, rl in self.sockets:
            for line in rl(s):
                try:
                    yield from self.protocol.decode(line)
                except Exception as e:
                    name = self.protocol.__name__
                    self.logger.debug(f"Protocol {name} decode error: {e}")

    def transform(self, item: P):
        # Only send to clients if TX is enabled
        if not self.TX:
            raise RuntimeError(f"TX disabled on {self}")
        # Encode and send a message
        protocol = self.protocol
        try:
            encoded = protocol.encode(item)
        except Exception as e:
            name = protocol.__name__
            self.logger.debug(f"Protocol {name} encode error: {e}")
            return
        if type(encoded) is str:
            msg = encoded
        elif isinstance(encoded, Iterable):
            msg = protocol.sep.join(encoded) + protocol.end
        else:
            logger.debug(f"Error handling protocol: {protocol.__name__}")
            logger.debug(f"Encoder produced: {encoded}")
            return
        # Broadcast outgoing message to all connections
        for s, rl in self.sockets:
            try:
                s.sendall(msg.encode())
            except TimeoutError:
                pass
            except OSError as e:
                err = e.args[0]
                if err in (errno.EAGAIN, errno.EWOULDBLOCK, errno.ETIMEDOUT):
                    pass
                else:
                    self.logger.debug(f"Failed to send message: {e}")
                    s.close()

    def __iter__(self):
        # Iterate through queued messages (never-ending)
        yield from self()

    def close(self):
        self.stop()
        for s, _ in self.sockets:
            s.close()


class Server(SocketTransport[P]):
    def __init__(self, path: Path, protocol: type[Protocol[P]] = DefaultProtocol, **kw):
        super().__init__(path, protocol, **kw)
        if self.path.exists():
            self.logger.debug(f"Socket {path} already exists")
            raise FileExistsError(path)
        self.server = socket(AF_UNIX, SOCK_STREAM)
        self.server.setblocking(False)
        self.server.bind(str(path))
        self.logger.debug(f"{self} started")

    def spin(self):
        # Accept all incoming connections
        while True:
            try:
                self.server.listen()
                client, _ = self.server.accept()
                client.settimeout(1e-3)
                self.sockets.add((client, Readlines()))
                self.logger.debug(f"{self} accepted new connection")
            except TimeoutError:
                break
            except OSError as e:
                err = e.args[0]
                if err in (errno.EAGAIN, errno.ETIMEDOUT):
                    break
                else:
                    raise e
        yield from super().spin()

    def __del__(self):
        with Expect():
            super().__del__()
        self.close()

    def close(self):
        super().close()
        if hasattr(self, "server"):
            no_throw(self.server.close)()
            no_throw(self.path.unlink)()
            del self.server
            self.logger.debug(f"{self} closed")


class Client(SocketTransport[P]):
    def __init__(self, path: Path, protocol: type[Protocol[P]] = DefaultProtocol, **kw):
        super().__init__(path, protocol, **kw)

    report_missing_socket = True
    last_attempt: float = time.time()
    retry_interval: float = 1.0

    def spin(self):
        # Normal operation
        if len(self.sockets):
            yield from super().spin()
        # Check if there is at least one active connection
        if len(self.sockets):
            return
        # Set the path for the Unix socket
        if not self.path.exists():
            if self.report_missing_socket:
                self.logger.debug(f"{self} waiting for server ...")
                self.report_missing_socket = False
            return
        else:
            self.report_missing_socket = True
        # Reconnect rate throttling
        now = time.time()
        should_retry = now - self.last_attempt > self.retry_interval
        if not should_retry:
            return
        else:
            self.last_attempt = now
        # Create the Unix socket server
        client = socket(AF_UNIX, SOCK_STREAM)
        # Set the socket to non-blocking mode
        client.setblocking(False)
        # Bind the socket to the path
        self.logger.debug(f"{self} trying to connect")
        try:
            client.connect(str(self.path))
            self.logger.debug(f"{self} connected")
            self.sockets.add((client, Readlines()))
            return
        except TimeoutError:
            self.logger.debug(f"{self} connection timed out")
        except OSError as e:
            match e.args[0]:
                case errno.ECONNREFUSED:
                    self.logger.debug(f"{self} connection refused")
                case errno.ENOENT:
                    self.logger.debug(f"{self} not exist")
                case _:
                    self.logger.debug(f"{self} failed to connect: {e}")
        client.close()


# Example usage
if __name__ == "__main__":
    import argparse, sys

    name = Path(__file__).stem
    parser = argparse.ArgumentParser()
    parser.add_argument("role", nargs="?", choices=["client", "server"])
    parser.add_argument("--path", default=f"/tmp/{name}.sock", type=str)
    args = parser.parse_args()
    path = Path(args.path)
    sock: SocketTransport[str]
    match str(args.role).lower():
        case "server":
            sock = Server(path).start()
        case "client":
            sock = Client(path).start()
        case _:
            logger.debug("Invalid role: " + args.role)
            sys.exit(1)
    sock(sys.stdout)
    with Expect(KeyboardInterrupt):
        for line in sys.stdin:
            sock.send(line)
        logger.debug("Reached end of file")
        sys.exit(0)
    print(file=sys.stderr)
    logger.debug("Received keyboard interrupt")

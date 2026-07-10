from __future__ import annotations

import socket


def main() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("0.0.0.0", 2222))
        server.listen()
        while True:
            connection, _address = server.accept()
            with connection:
                connection.sendall(b"FILIN-LAB-ADMIN-SIM\n")


if __name__ == "__main__":
    main()

'''
Copied from https://github.com/Physical-Intelligence/openpi/blob/95aadc6b16d170e4b13ab2e0ac64fbf2d1bb8e31/packages/openpi-client/src/openpi_client/websocket_client_policy.py

Modified to fit our setup.
'''

import logging
import time
import websockets.sync.client

from typing import Dict, Optional, Tuple
from utils import msgpack_numpy

class WebsocketClientPolicy:
    """Implements the Policy interface by communicating with a server over websocket.

    See WebsocketPolicyServer for a corresponding server implementation.
    """

    def __init__(self, host: str = "0.0.0.0", port: Optional[int] = None, api_key: Optional[str] = None, connect_timeout: Optional[float] = None) -> None:
        self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = msgpack_numpy.Packer()
        self._api_key = api_key
        self._connect_timeout = connect_timeout
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> Dict:
        return self._server_metadata

    def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        logging.info(f"Waiting for server at {self._uri}...")
        started = time.monotonic()
        while True:
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                import inspect
                header_name = "additional_headers" if "additional_headers" in inspect.signature(websockets.sync.client.connect).parameters else "extra_headers"
                conn = websockets.sync.client.connect(
                    self._uri, compression=None, max_size=None, **{header_name: headers}
                )
                metadata = msgpack_numpy.unpackb(conn.recv())
                return conn, metadata
            except ConnectionRefusedError:
                if self._connect_timeout is not None and time.monotonic() - started >= self._connect_timeout:
                    raise TimeoutError("model server did not become available")
                logging.info("Still waiting for server...")
                time.sleep(5)

    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        data = self._packer.pack(obs)
        self._ws.send(data)
        response = self._ws.recv()
        if isinstance(response, str):
            # we're expecting bytes; if the server sends a string, it's an error.
            raise RuntimeError(f"Error in inference server:\n{response}")
        return msgpack_numpy.unpackb(response)

    def close(self):
        self._ws.close()

import os
import sys
import json
config = {
    "mcpServers": {
        "study-server": {
            "command": sys.executable,                       # the Python this notebook is using
            "args": [os.path.abspath("testmcp.py")],    # the full path to the server file
        }
    }
}
print(json.dumps(config, indent=2))

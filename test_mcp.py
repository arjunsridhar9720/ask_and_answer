from fastmcp.client import Client
from fastmcp.client.transports import StdioTransport
import os
import asyncio
 
#  mongodb+srv://chatgpt-openai-dev-pl-0.nmkkl.mongodb.net/?authMechanism=MONGODB-X509&authSource=%24external&tls=true&tlsCertificateKeyFile=C%3A%5CUsers%5CArjun.Sridharkumar%5CDownloads%5CX509-chatctc-app.pem
# Set environment variables
os.environ["MONGODB_URI"] = "mongodb+srv://mlab-029-itgm-stage-cc-pl-0.bcrga.mongodb.net"
os.environ["MONGODB_CERT_PATH"] = "C:/Users/evan.sinukoff/Projects/repos/ignite-search-repo/app/config/corpstage029mlab-kv-mongodbatlas-mlab-029-itgm-stage-cc-svc-rw-pem-20240906.pem"
 
command = "npx"
args = [
    "-y",
    "mongodb-mcp-server",
    "--uri", os.environ["MONGODB_URI"],
    "--tlsCertificateKeyFile", os.environ["MONGODB_CERT_PATH"],
    "--authMechanism", "MONGODB-X509",
    "--authSource", "$external",
    "--tls", "true",
    "--tlsAllowInvalidCertificates", "false",
    "--directConnection", "false",
    "--tlsAllowInvalidHostnames", "true"
]
 
client = Client(StdioTransport(command, args))
 
async def main():
    async with Client(StdioTransport(command, args)) as client:
        # Connect to MongoDB
        await client.call_tool("connect", {
        "connectionString": os.environ["MONGODB_URI"] +
            "?tls=true"
            "&tlsCertificateKeyFile=" + os.environ["MONGODB_CERT_PATH"] +
            "&authMechanism=MONGODB-X509"
            "&authSource=$external"
            "&tlsAllowInvalidCertificates=false"
            "&directConnection=false"
            "&tlsAllowInvalidHostnames=true"
    })
        # Now list databases
        result = await client.call_tool("list-databases", {})
        print(result)
 
asyncio.run(main())
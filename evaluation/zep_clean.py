from zep_cloud.client import AsyncZep
import asyncio

client =AsyncZep(api_key="z_1dWlkIjoiOGQ5NThjYWUtNDdlOC00ZWE3LWE0OTgtZGVmNGE4NTUwN2Q0In0.UU5xkvc3iRK4Z74vXaQGSMO_0TM4IETV5LSybKVy5Cq-uYCS_CMeUwN9UDbJFwj-2USfzhFq6Gxk41o_dIU7IQ")

async def delete():
    await client.graph.delete(graph_id="mobilemem_0")
    print("Done")

asyncio.run(delete())
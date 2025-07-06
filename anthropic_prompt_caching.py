from pydantic import BaseModel, Field
from typing import Optional

class Filter:
    class Valves(BaseModel):
        pass

    def __init__(self):
        self.valves = self.Valves()
        self.toggle = True # IMPORTANT: This creates a switch UI in Open WebUI
        pass

    async def inlet(
        self, body: dict, __event_emitter__, __user__: Optional[dict] = None
    ) -> dict:
        await __event_emitter__(
            {
                "type": "status",
                "data": {
                    "description": "Prompt Caching Enabled!",
                    "done": True,
                    "hidden": False,
                },
            }
        )
        print("Prompt Caching Enabled!")
        last_message_item = body["messages"][-1]
        # 如果 content 是字符串
        if isinstance(last_message_item["content"], str):
            last_message_item['content'] = [{"type": "text", "text": last_message_item["content"], "cache_control": {"type": "ephemeral"}}]
        else:
            print("Prompt Caching Failed: unsupported content type")
        return body
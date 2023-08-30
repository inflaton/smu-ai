"""Main entrypoint for the app."""
import json
import os
from timeit import default_timer as timer
from typing import List, Optional

from lcserve import serving
from pydantic import BaseModel

from app_modules.init import app_init
from app_modules.llm_chat_chain import ChatChain
from app_modules.utils import print_llm_response

llm_loader, qa_chain = app_init()

chat_history_enabled = os.environ.get("CHAT_HISTORY_ENABLED") == "true"

uuid_to_chat_chain_mapping = dict()


class ChatResponse(BaseModel):
    """Chat response schema."""

    token: Optional[str] = None
    error: Optional[str] = None
    sourceDocs: Optional[List] = None


def do_chat(
    question: str,
    history: Optional[List] = None,
    chat_id: Optional[str] = None,
    streaming_handler: any = None,
):
    if history is not None:
        chat_history = []
        if chat_history_enabled:
            for element in history:
                item = (element[0] or "", element[1] or "")
                chat_history.append(item)

        start = timer()
        result = qa_chain.call_chain(
            {"question": question, "chat_history": chat_history, "chat_id": chat_id},
            streaming_handler,
        )
        end = timer()
        print(f"Completed in {end - start:.3f}s")

        print(f"qa_chain result: {result}")
        return result
    else:
        if chat_id in uuid_to_chat_chain_mapping:
            chat = uuid_to_chat_chain_mapping[chat_id]
        else:
            chat = ChatChain(llm_loader)
            uuid_to_chat_chain_mapping[chat_id] = chat
        result = chat.call_chain({"question": question}, streaming_handler)
        print(f"chat result: {result}")
        return result


@serving(websocket=True)
def chat(
    question: str,
    history: Optional[List] = None,
    chat_id: Optional[str] = None,
    **kwargs,
) -> str:
    print("question@chat:", question)
    streaming_handler = kwargs.get("streaming_handler")
    result = do_chat(question, history, chat_id, streaming_handler)
    resp = ChatResponse(
        sourceDocs=result["source_documents"] if history is not None else []
    )
    return json.dumps(resp.dict())


@serving
def chat_sync(
    question: str,
    history: Optional[List] = None,
    chat_id: Optional[str] = None,
    **kwargs,
) -> str:
    print("question@chat_sync:", question)
    result = do_chat(question, history, chat_id, None)
    return result["response"]


if __name__ == "__main__":
    # print_llm_response(json.loads(chat("What's deep learning?", [])))
    chat_start = timer()
    chat_sync("what's deep learning?", chat_id="test_user")
    chat_sync("more on finance", chat_id="test_user")
    chat_sync("more on Sentiment analysis", chat_id="test_user")
    chat_sync("Write the game 'snake' in python", chat_id="test_user")
    # chat_sync("给我讲一个年轻人奋斗创业最终取得成功的故事。", chat_id="test_user")
    # chat_sync("给这个故事起一个标题", chat_id="test_user")
    chat_end = timer()
    total_time = chat_end - chat_start
    print(f"Total time used: {total_time:.3f} s")
    print(f"Number of tokens generated: {llm_loader.streamer.total_tokens}")
    print(
        f"Average generation speed: {llm_loader.streamer.total_tokens / total_time:.3f} tokens/s"
    )

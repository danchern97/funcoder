from utils import UnitTestConfig, async_test_case

from funcoder.llm.config import create_llm_engine


@async_test_case
async def test_llm_openai_gpt_engine():
    UTC = UnitTestConfig()
    if not UTC.test_llm():
        return
    config = UTC.mk_llm_config()
    engine = create_llm_engine(config)
    response = await engine.call([{"role": "user", "content": "Hello, how are you?"}], n=4)
    assert response.status == "ok"
    assert len(response.ok) == 4
    assert len(response.ok[0]) > 0
    assert len(response.ok[1]) > 0

import asyncio

from funcoder.utils.pyctx import PyCtx


def test_utils_pyctx_working():
    """
    > [ a ] ---+-->   b   ------> [ c ] ------> [ d ]
    >  ^^^     |     ^^^           ^^^
    >          +----------------> [[e]] ------> [ f ]
    """

    ctx = PyCtx[str]("test_working")
    result: list[list[str]] = []

    async def a() -> None:
        result.append(ctx.get())
        ctx.append("AAA")
        result.append(ctx.get())
        await b()
        result.append(ctx.get())
        e()
        result.append(ctx.get())

    async def b() -> None:
        result.append(ctx.get())
        await c()
        result.append(ctx.get())

    async def c() -> None:
        result.append(ctx.get())
        ctx.append("CCC")
        result.append(ctx.get())
        d()
        result.append(ctx.get())

    def d() -> None:
        result.append(ctx.get())
        ctx.append("DDD")
        result.append(ctx.get())

    def e() -> None:
        result.append(ctx.get())
        ctx.update(lambda x: "EE1" if not x else x[0] + "EE0")
        result.append(ctx.get())
        ctx.append("EE2")
        result.append(ctx.get())
        ctx.update(lambda x: "EEE" if not x else x[0] + "EE3")
        result.append(ctx.get())
        f()
        result.append(ctx.get())

    def f() -> None:
        result.append(ctx.get())
        ctx.append("FFF")
        result.append(ctx.get())

    expected: list[list[str]] = [
        [],  # a_0
        ["AAA"],  # a_1
        ["AAA"],  # b_0
        ["AAA"],  # c_0
        ["AAA", "CCC"],  # c_1
        ["AAA", "CCC"],  # d_0
        ["AAA", "CCC", "DDD"],  # d_1
        ["AAA", "CCC"],  # c_2
        ["AAA"],  # b_1
        ["AAA"],  # a_2
        ["AAA"],  # e_0
        ["AAA", "EE1"],  # e_1
        ["AAA", "EE2"],  # e_2
        ["AAA", "EE2EE3"],  # e_3
        ["AAA", "EE2EE3"],  # f_0
        ["AAA", "EE2EE3", "FFF"],  # f_1
        ["AAA", "EE2EE3"],  # e_3
        ["AAA"],  # a_3
    ]
    asyncio.run(a())
    assert result == expected

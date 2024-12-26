from funcoder.utils.treestore import TreeStore


def test_working():
    store = TreeStore[str]("_test")

    def a() -> list[str]:
        _ = store.trap()
        store.put("AAA")
        trap = store.trap()
        b(2)
        return trap.gather()

    def b(cnt: int) -> None:
        trap = store.trap()
        c(cnt * 2)
        c(cnt * 3)
        store.put(",".join(trap.gather()))

    def c(cnt: int) -> None:
        store.put("c" * cnt)
        _trap = store.trap()
        store.put("CCC")

    assert a() == ["cccc,cccccc"]

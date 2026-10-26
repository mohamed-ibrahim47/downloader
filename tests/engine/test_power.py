from downloader.engine.power import sleep_pc


def test_sleep_pc_calls_runner():
    calls = []
    sleep_pc(runner=lambda cmd, **kw: calls.append(cmd))
    assert calls == [["rundll32.exe", "powrprof.dll,SetSuspendState", "0,1,0"]]

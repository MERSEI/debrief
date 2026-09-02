from debrief.stt.devices import CPU_COMPUTE, GPU_COMPUTE, build_plan


class TestAuto:
    def test_prefers_gpu_but_keeps_cpu_as_fallback(self):
        # Видимый GPU может оказаться непригодным: cuBLAS ставится отдельно
        # от драйвера, поэтому вторая попытка обязательна.
        assert build_plan("auto", "default", cuda_available=True) == [
            ("cuda", GPU_COMPUTE),
            ("cpu", CPU_COMPUTE),
        ]

    def test_without_gpu_goes_straight_to_cpu(self):
        assert build_plan("auto", "default", cuda_available=False) == [
            ("cpu", CPU_COMPUTE)
        ]

    def test_explicit_compute_type_is_kept_across_the_plan(self):
        assert build_plan("auto", "int8", cuda_available=True) == [
            ("cuda", "int8"),
            ("cpu", "int8"),
        ]


class TestExplicitDevice:
    def test_named_device_never_falls_back(self):
        # Написал cuda — хочешь знать, что она не работает, а не получить
        # молча десятикратно более медленный прогон на процессоре.
        assert build_plan("cuda", "default", cuda_available=True) == [
            ("cuda", GPU_COMPUTE)
        ]

    def test_named_cpu_gets_int8_by_default(self):
        assert build_plan("cpu", "default", cuda_available=True) == [
            ("cpu", CPU_COMPUTE)
        ]

    def test_named_device_respects_explicit_compute_type(self):
        assert build_plan("cuda", "float32", cuda_available=False) == [
            ("cuda", "float32")
        ]

from debrief.audio.timeline import missing_frames, silence

RATE = 48000


class TestMissingFrames:
    def test_silent_device_gets_padded_to_real_time(self):
        # Пять секунд ничего не приходило — ровно то, что делает loopback,
        # пока через устройство вывода не играет звук.
        assert missing_frames(5.0, RATE, frames_written=0) == 5 * RATE

    def test_stream_in_sync_needs_nothing(self):
        assert missing_frames(1.0, RATE, frames_written=RATE) == 0

    def test_jitter_within_tolerance_is_ignored(self):
        # Отстаём на 50 мс при допуске 100 мс — это дрожание буферов.
        written = RATE - int(0.05 * RATE)
        assert missing_frames(1.0, RATE, frames_written=written) == 0

    def test_gap_beyond_tolerance_is_filled(self):
        written = RATE - int(0.5 * RATE)
        assert missing_frames(1.0, RATE, frames_written=written) == int(0.5 * RATE)

    def test_stream_ahead_of_clock_is_never_trimmed(self):
        # Пришло больше, чем ожидалось: резать уже записанное нельзя.
        assert missing_frames(1.0, RATE, frames_written=2 * RATE) == 0

    def test_zero_elapsed_is_a_noop(self):
        assert missing_frames(0.0, RATE, frames_written=0) == 0

    def test_negative_elapsed_is_a_noop(self):
        assert missing_frames(-1.0, RATE, frames_written=0) == 0

    def test_rate_of_each_track_is_respected(self):
        # Микрофон 44100, loopback 48000 — пропуск считается в своих фреймах.
        assert missing_frames(2.0, 44100, frames_written=0) == 88200
        assert missing_frames(2.0, 48000, frames_written=0) == 96000


class TestSilence:
    def test_block_size_matches_format(self):
        assert len(silence(100, channels=2, sample_width=2)) == 100 * 2 * 2

    def test_block_is_digital_zero(self):
        assert set(silence(10, 1, 2)) == {0}

    def test_no_frames_is_empty(self):
        assert silence(0, 2, 2) == b""
        assert silence(-5, 2, 2) == b""

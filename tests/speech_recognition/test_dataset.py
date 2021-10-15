import os
import unittest
from io import StringIO, BytesIO
from typing import Dict
from unittest.mock import patch, call, MagicMock

import numpy
import soundfile
from torch import Tensor

from fairseq.data import FileAudioDataset
from fairseq.data.audio.audio_utils import parse_path, ParsedPath
from fairseq.data.audio.speech_to_text_dataset import get_features_or_waveform

SAMPLE_RATE = 16000

_soundfile_read = soundfile.read


class TestDataset(unittest.TestCase):

    @patch("os.path.isfile", return_value=True)
    def test_parse_path(self, mock_is_file):
        assert os.path.isfile is mock_is_file

        test_cases = [
            ("/path/to/myfile.wav:123:456", ParsedPath("/path/to/myfile.wav", start=123, frames=456)),
            ("/path/to/myfile.zip:123:456", ParsedPath("/path/to/myfile.zip", zip_offset=123, zip_length=456)),
        ]

        for path, expected in test_cases:
            with self.subTest():
                parsed = parse_path(path)
                self.assertEqual(parsed, expected)

    def test_file_audio_dataset_audio_with_offsets_etc(self):
        expected_soundfile_reads = [
            call("/path/to/myfile1.wav", frames=456, start=123, dtype="float32"),
            call("/path/to/myfile2.wav", frames=-1, start=0, dtype="float32"),
        ]

        files = {
            "/dev/null/train.tsv": os.linesep.join([
                "/dev/null",
                "/path/to/myfile1.wav:123:456\t456",
                "/path/to/myfile2.wav\t456",
                ""
            ]).encode(),
            "/path/to/myfile1.wav": random_wav_data(size=456, pad_wav=122),
            "/path/to/myfile2.wav": random_wav_data(size=456,),
            "/path/to/myfile1.zip": random_wav_data(size=456, pad_wav=122),
            "/path/to/myfile2.zip": random_wav_data(size=456),
        }

        with MockSoundFiles(files) as msf, self.subTest():
            dataset = FileAudioDataset("/dev/null/train.tsv", sample_rate=SAMPLE_RATE)
            self.assertEqual(len(dataset.fnames), len(expected_soundfile_reads))
            dataset_iter = iter(dataset)

            for n, expected_call in enumerate(expected_soundfile_reads):
                with self.subTest(dataset.fnames[n]):
                    item = next(dataset_iter)
                    self.assertIsInstance(item["source"], Tensor)

                with self.subTest("soundfile.read() *args"):
                    assert msf.mock_soundfile_read.call_count == n + 1
                    actual_call = msf.mock_soundfile_read.call_args
                    self.assertEqual(expected_call.args, actual_call.args)

                with self.subTest("soundfile.read() **kwargs"):
                    self.assertEqual(expected_call.kwargs, actual_call.kwargs)

    def test_file_audio_dataset_zip_with_offsets_etc(self):
        expected_calls = [            # open(), f.seek(), f.read(), soundfile.read()
            (
                call("/path/to/myfile1.zip", "rb"), call(123), call(456),
                call("/path/to/myfile1.zip", frames=-1, start=0, dtype="float32")
            ),
        ]

        files = {
            "/dev/null/train.tsv": os.linesep.join([
                "/dev/null",
                "/path/to/myfile1.zip:123:456\t456",
            ]).encode(),
            "/path/to/myfile1.zip": random_wav_data(456, pad_zip=123)
        }

        with MockSoundFiles(files) as msf, self.subTest():
            dataset = FileAudioDataset("/dev/null/train.tsv", sample_rate=SAMPLE_RATE)
            self.assertEqual(len(dataset.fnames), len(expected_calls))
            dataset_iterator = iter(dataset)

            for n, (e_open_call, e_seek_call, e_read_call, e_sf_call) in enumerate(expected_calls):
                with self.subTest(dataset.fnames[n]):
                    item = next(dataset_iterator)
                    self.assertIsInstance(item["source"], Tensor)

                with self.subTest("open() call"):
                    self.assertEqual(msf.mock_open.call_args, e_open_call)

                with self.subTest("f.seek() call"):
                    actual_seek_call = msf.mock_open.handles[-1].seek.call_args
                    self.assertEqual(actual_seek_call, e_seek_call)

                with self.subTest("f.read() call"):
                    actual_read_call = msf.mock_open.handles[-1].read.call_args
                    self.assertEqual(actual_read_call, e_read_call)

                with self.subTest("soundfile.read() **kwargs"):
                    assert msf.mock_soundfile_read.call_count == n + 1
                    actual_sf_read_call = msf.mock_soundfile_read.call_args
                    self.assertEqual(actual_sf_read_call.kwargs, e_sf_call.kwargs)

    def test_get_features_or_waveform(self):
        test_cases = [
            (
                "/path/to/myfile1.wav", {"need_waveform": True},
                random_wav_data(size=456),
                (None, None, None),
                call("/path/to/myfile1.wav", frames=-1, start=0, always_2d=True, dtype="float32"),
            ),
            (
                "/path/to/myfile1.wav:123:456", {"need_waveform": True},
                random_wav_data(size=456, pad_wav=123),
                (None, None, None),
                call("/path/to/myfile1.wav", frames=456, start=123, always_2d=True, dtype="float32"),
            ),
            (
                "/path/to/myfile1.wav:123:456", {"need_waveform": False},
                random_wav_data(size=456, pad_wav=123),
                (None, None, None),
                call("/path/to/myfile1.wav", frames=456, start=123, always_2d=True, dtype="float32"),
            ),
            (
                "/path/to/myfile1.zip", {"need_waveform": True},
                random_wav_data(size=410),
                (call("/path/to/myfile1.zip", "rb"), call(0), call(-1)),
                call("/path/to/myfile1.zip", frames=-1, start=0, always_2d=True, dtype="float32"),
            ),
            (
                "/path/to/myfile2.zip:123:456", {"need_waveform": True},
                random_wav_data(size=456, pad_zip=123),
                (call("/path/to/myfile2.zip", "rb"), call(123), call(456)),
                call("/path/to/myfile2.zip", frames=-1, start=0, always_2d=True, dtype="float32"),
            ),
        ]

        for filename_unparsed, kwargs, wav_data, (e_open_call, e_seek_call, e_read_call), e_sf_read_call in test_cases:
            filename = filename_unparsed.split(":")[0]
            with MockSoundFiles({filename: wav_data}) as msf:
                with self.subTest(filename_unparsed):
                    features = get_features_or_waveform(filename_unparsed, **kwargs)
                    self.assertIsInstance(features, numpy.ndarray)

                with self.subTest("open() call"):
                    self.assertEqual(msf.mock_open.call_args, e_open_call)

                with self.subTest("f.seek() call"):
                    if len(msf.mock_open.handles):
                        actual_seek_call = msf.mock_open.handles[-1].seek.call_args
                        self.assertEqual(actual_seek_call, e_seek_call)
                    else:
                        self.assertEqual(None, e_seek_call)

                with self.subTest("f.read() call"):
                    if len(msf.mock_open.handles):
                        actual_read_call = msf.mock_open.handles[-1].read.call_args
                        self.assertEqual(actual_read_call, e_read_call)
                    else:
                        self.assertEqual(None, e_read_call)

                with self.subTest(f"soundfile.read() **kwargs"):
                    assert msf.mock_soundfile_read.call_count == 1
                    actual_sf_read_call = msf.mock_soundfile_read.call_args
                    self.assertEqual(e_sf_read_call.kwargs, actual_sf_read_call.kwargs)


class MockSoundFiles(Dict[str, bytes]):
    def __enter__(self):
        self.mock_soundfile_read = patch("soundfile.read", side_effect=self.soundfile_read).__enter__()
        self.mock_isfile = patch("os.path.isfile", return_value=True).__enter__()
        self.mock_open = patch("builtins.open", side_effect=self.open).__enter__()
        self.mock_open.handles = []
        assert open is self.mock_open
        assert os.path.isfile is self.mock_isfile
        assert soundfile.read is self.mock_soundfile_read
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.mock_soundfile_read.__exit__()
        self.mock_isfile.__exit__()
        self.mock_open.__exit__()
        return exc_type is None

    def open(self, file, mode='r', buffering=None, encoding=None, errors=None, newline=None, closefd=True):
        data = self[file]
        if 'b' in mode:
            data = BytesIO(data)
        else:
            data = StringIO(data.decode(encoding or "utf8"))
        mock_handle = MagicMock(wraps=data)
        mock_handle.__enter__.return_value = mock_handle
        mock_handle.__iter__.side_effect = lambda: iter(data)
        self.mock_open.handles.append(mock_handle)
        return mock_handle

    def soundfile_read(self, file, *args, **kwargs):
        if isinstance(file, str):
            file = BytesIO(self[file])
        return _soundfile_read(BytesIO(file.getvalue()), *args, **kwargs)


def random_wav_data(size, sample_rate=SAMPLE_RATE, pad_zip=0, pad_wav=0):
    bytes_io = BytesIO()
    x = numpy.zeros((pad_wav + size))
    x[pad_wav:] = numpy.random.rand((size)).astype("float32")
    soundfile.write(bytes_io, x, samplerate=sample_rate, format="wav")
    pad = bytes(ord("0") + (n % 10) for n in range(pad_zip))
    return pad + bytes_io.getvalue()


if __name__ == '__main__':
    unittest.main()

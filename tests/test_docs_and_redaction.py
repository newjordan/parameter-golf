import json

import numpy as np
import pytest

from data.download_hf_docs_and_tokenize import PureByteTokenizer, count_docs, iter_docs
from parameter_golf.redact import is_sensitive_path, redact_text
from parameter_golf.shards import read_shard, write_shard


def test_iter_docs_skips_blank_lines_and_reports_bad_json(tmp_path):
    path = tmp_path / "docs.jsonl"
    path.write_text(
        '\n{"text": "alpha"}\n\n{"text": "beta"}\n',
        encoding="utf-8",
    )
    assert list(iter_docs(path)) == ["alpha", "beta"]
    assert count_docs(path) == 2
    path.write_text('{"text": "ok"}\nnot-json\n', encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        list(iter_docs(path))


def test_pure_byte_tokenizer_uses_a_real_uint16_cast():
    encoded = PureByteTokenizer().encode("Aé")
    assert encoded.dtype == np.dtype(np.uint16)
    assert encoded.tolist() == [ord("A") + 4, 0xC3 + 4, 0xA9 + 4]


def test_written_docs_shard_can_be_read_back(tmp_path):
    tokens = PureByteTokenizer().encode("hi")
    path = tmp_path / "shard.bin"
    write_shard(path, tokens)
    assert read_shard(path).tolist() == tokens.tolist()


def test_redaction_removes_instance_keys_ssh_urls_and_env_files(tmp_path):
    leaked = (
        "Started. {'instance_api_key': '0123456789abcdef0123456789abcdef'} "
        "SSH_URL=ssh://root@example.invalid:22"
    )
    cleaned = redact_text(leaked)
    assert "0123456789abcdef0123456789abcdef" not in cleaned
    assert "ssh://root@example.invalid:22" not in cleaned
    assert "[redacted]" in cleaned
    assert is_sensitive_path(tmp_path / "ssh_info.env")
    assert is_sensitive_path(tmp_path / "id_rsa")
    assert not is_sensitive_path(tmp_path / "submission.json")


def test_submission_json_roundtrip_helper_is_not_required():
    payload = json.loads('{"val_bpb": 1.1}')
    assert payload["val_bpb"] == 1.1

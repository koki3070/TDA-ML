"""Tests for ellphi_repo pin / ensure reproducibility helpers."""

from __future__ import annotations

import subprocess
import unittest
from pathlib import Path

from tda_ml.reproducibility import (
    ELLPHI_REPO_URL,
    assert_ellphi_differentiable_available,
    assert_ellphi_repo_matches_pin,
    build_ellphi_repo_manifest_fields,
    read_ellphi_repo_head,
    read_pinned_ellphi_revision,
)

REPO = Path(__file__).resolve().parents[1]


class TestEllphiRepoPin(unittest.TestCase):
    def test_ref_file_has_40_char_sha(self) -> None:
        pinned = read_pinned_ellphi_revision(REPO)
        self.assertIsNotNone(pinned)
        self.assertEqual(len(pinned), 40)
        self.assertTrue(all(c in "0123456789abcdef" for c in pinned.lower()))

    def test_manifest_fields_include_pin(self) -> None:
        fields = build_ellphi_repo_manifest_fields(REPO)
        self.assertEqual(fields["ellphi_repo_url"], ELLPHI_REPO_URL)
        self.assertIn("ellphi_repo_revision_pinned", fields)

    def test_installed_matches_pin_when_ellphi_repo_present(self) -> None:
        if read_ellphi_repo_head(REPO) is None:
            self.skipTest("ellphi_repo not checked out")
        assert_ellphi_repo_matches_pin(project_root=REPO)
        fields = build_ellphi_repo_manifest_fields(REPO)
        self.assertFalse(fields.get("ellphi_repo_revision_mismatch", True))

    def test_grad_api_available_when_ellphi_repo_present(self) -> None:
        if read_ellphi_repo_head(REPO) is None:
            self.skipTest("ellphi_repo not checked out")
        impl = assert_ellphi_differentiable_available(ellphi_differentiable=True)
        self.assertEqual(impl, "ellphi_torch_grad")

    def test_ensure_script_is_executable(self) -> None:
        script = REPO / "scripts" / "ensure_ellphi_repo.sh"
        self.assertTrue(script.is_file())
        mode = script.stat().st_mode
        self.assertTrue(mode & 0o111, "ensure_ellphi_repo.sh must be executable")

    def test_ensure_script_idempotent(self) -> None:
        script = REPO / "scripts" / "ensure_ellphi_repo.sh"
        if read_ellphi_repo_head(REPO) is None:
            self.skipTest("ellphi_repo not checked out")
        subprocess.run([str(script)], cwd=REPO, check=True, capture_output=True, text=True)


if __name__ == "__main__":
    unittest.main()

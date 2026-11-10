# SPDX-License-Identifier: Apache-2.0
"""Web-page generator checks; they never write into the package."""

import contextlib
import gzip
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest


TOOLS = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("rlcd_build_web", TOOLS / "build_web.py")
web = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(web)


def quiet_check(source, target):
    with contextlib.redirect_stdout(io.StringIO()) as output:
        code = web.check(source, target)
    return code, output.getvalue()


class BuildWebTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="rlcd-web-test-")
        self.root = Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def test_committed_header_is_a_fresh_build_of_index_html(self):
        minified, data, text = web.build(web.SOURCE.read_text(encoding="utf-8"))
        self.assertEqual(gzip.decompress(data), minified)
        self.assertLess(len(data), web.SIZE_LIMIT)
        self.assertEqual(text.encode("ascii"), web.TARGET.read_bytes())
        self.assertEqual(quiet_check(web.SOURCE, web.TARGET)[0], 0)

    def test_check_fails_for_an_edited_page_or_a_missing_header(self):
        page = self.root / "index.html"
        page.write_text(web.SOURCE.read_text(encoding="utf-8").replace("</body>", "<p>edited</p></body>", 1),
                        encoding="utf-8")
        code, output = quiet_check(page, web.TARGET)
        self.assertEqual(code, 1)
        self.assertIn("does not match", output)
        self.assertEqual(quiet_check(web.SOURCE, self.root / "missing.h")[0], 1)

    def test_check_accepts_the_same_page_compressed_by_another_zlib(self):
        minified, _, _ = web.build(web.SOURCE.read_text(encoding="utf-8"))
        other = gzip.compress(minified, compresslevel=1, mtime=0)
        target = self.root / "WebPage.h"
        target.write_bytes(web.header(other).encode("ascii"))
        code, output = quiet_check(web.SOURCE, target)
        self.assertEqual(code, 0)
        self.assertIn("gzip stream differs", output)
        target.write_bytes(web.header(other).replace("Do not edit by hand.", "Edited.").encode("ascii"))
        self.assertEqual(quiet_check(web.SOURCE, target)[0], 1)

    def test_minifier_refuses_constructs_it_would_change(self):
        for page in ("<script>\nlet a = `x`;\n</script>", "<pre>\n  kept\n</pre>",
                     "<script>\nlet a = 1 + \\\n2;\n</script>", "<script>\n/* a */ code();\n</script>"):
            with self.subTest(page=page), self.assertRaises(web.PageError):
                web.build(page)


if __name__ == "__main__":
    unittest.main()

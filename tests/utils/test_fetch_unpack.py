# coding=utf-8
#
# created by kpe on 09.Aug.2019 at 15:28
#

from __future__ import absolute_import, division, print_function

import unittest

import tempfile

import params_flow as pf


FETCH_URL = "https://files.pythonhosted.org/packages/4c/2a/79f44178ac6f5b6995bc7804898fce2654da70e0c5c7f76332748420d6fd/bert-for-tf2-0.13.5.tar.gz"
FETCH_ZIP_URL = "https://github.com/kpe/bert-for-tf2/archive/v0.13.5.zip"


class UtilsTest(unittest.TestCase):
    def test_fetch_and_unpack_targz(self):

        with tempfile.TemporaryDirectory() as tmp_dir:
            fetched_file = pf.utils.fetch_url(FETCH_URL, tmp_dir)
            fetched_file = pf.utils.fetch_url(FETCH_URL, tmp_dir)
            fetched_file = pf.utils.fetch_url(FETCH_URL, tmp_dir, check_content_length=True)
            fetched_dir = pf.utils.unpack_archive(fetched_file)
            fetched_dir = pf.utils.unpack_archive(fetched_file)

        with tempfile.TemporaryDirectory() as tmp_dir:
            fetched_file = pf.utils.fetch_url(FETCH_ZIP_URL, tmp_dir)
            fetched_dir = pf.utils.unpack_archive(fetched_file)

        try:
            pf.utils.unpack_archive(fetched_file+".mp4")
        except:
            pass

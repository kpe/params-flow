# coding=utf-8
#
# created by kpe on 09.Aug.2019 at 15:28
#

from __future__ import absolute_import, division, print_function

import unittest

import tempfile

import params_flow as pf


FETCH_URL = "https://files.pythonhosted.org/packages/06/60/6c28bf004c8b6706d3434d253d7e455ba63a63d0307ab0abe500f23af934/bert-for-tf2-0.3.7.tar.gz"
FETCH_ZIP_URL = "https://github.com/kpe/bert-for-tf2/archive/v0.3.7.zip"


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

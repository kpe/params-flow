# coding=utf-8
#
# created by kpe on 09.Aug.2019 at 15:28
#

from __future__ import absolute_import, division, print_function

import unittest

import tempfile

import params_flow as pf


FETCH_URL = "https://files.pythonhosted.org/packages/d3/89/ee123cc61809e7400b52554b8620e1f4a69dde83aede8349fff5c5ae1539/bert-for-tf2-0.4.2.tar.gz"
FETCH_ZIP_URL = "https://github.com/kpe/bert-for-tf2/archive/v0.4.2.zip"


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

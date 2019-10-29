# coding=utf-8
#
# created by kpe on 09.Aug.2019 at 15:26
#

from __future__ import absolute_import, division, print_function

import os
import re
import urllib


from tqdm import tqdm


def fetch_url(url, fetch_dir, check_content_length=False, local_file_name=None):
    """
    Downloads the specified url to a local dir.
    :param url:
    :param fetch_dir:
    :param check_content_length:
    :param local_file_name:
    :return: local path of the downloaded file
    """
    if local_file_name is None:
        url_path = urllib.parse.urlparse(url).path
        local_file_name = url_path.split('/')[-1]
    local_path = os.path.join(fetch_dir, local_file_name)

    already_fetched = False
    if os.path.isfile(local_path):
        if check_content_length:
            content_length = int(urllib.request.urlopen(url).getheader("Content-Length"))
            already_fetched = os.stat(local_path).st_size == content_length
        else:
            already_fetched = True

    if already_fetched:
        print("Already  fetched: ", local_file_name)
    else:
        os.makedirs(fetch_dir, exist_ok=True)
        with tqdm(unit='B', unit_scale=True,
                  miniters=1, desc=local_file_name) as pbar:
            def report_hook(count, block_size, total_size):
                if total_size:
                    pbar.total = total_size
                pbar.update(block_size)
            urllib.request.urlretrieve(url, local_path, report_hook, data=None)
    return local_path


def unpack_archive(archive_file, unpack_dir=None):
    """ Unpacks a zip or a tar.{gz,bz2,xz} into the given dir.
    :param: zip_dir - if None unpacks in a dir with the same name as the zip file.
    """
    re_ext = re.compile(r'(\.zip|\.tar\.(?:gz|bz2|xz))$')

    archive_ext = re_ext.findall(archive_file)
    if len(archive_ext) < 1:
        raise AttributeError("Expecting .zip or tar.gz file, but: [{}]".format(archive_file))
    archive_ext = archive_ext[0]

    if unpack_dir is None:
        unpack_dir = os.path.basename(archive_file)
        unpack_dir = unpack_dir[:unpack_dir.rindex(archive_ext)]
        unpack_dir = os.path.join(os.path.dirname(archive_file), unpack_dir)

    if os.path.isdir(unpack_dir):
        print("already unpacked at: {}".format(unpack_dir))
    else:
        print("extracting to: {}".format(unpack_dir))
        if archive_ext == ".zip":
            import zipfile
            with zipfile.ZipFile(archive_file, "r") as zf:
                zf.extractall(unpack_dir)
        else:
            import tarfile
            with tarfile.open(archive_file, "r:*") as zf:
                zf.extractall(unpack_dir)

    return unpack_dir

# RoboMaster SDK offline Windows bundle

These files keep the classroom setup reproducible if DJI's legacy SDK repository or
the old PyPI wheel disappears. The setup scripts verify the files they execute or
install. Do not replace a file without updating and independently reviewing its hash.

| File | Original source | SHA256 | Setup behavior |
| --- | --- | --- | --- |
| `windows/python-3.8.10-amd64.exe` | `https://www.python.org/ftp/python/3.8.10/python-3.8.10-amd64.exe` | `7628244CB53408B50639D2C1287C659F4E29D3DFDB9084B11AED5870C0C6A48A` | Signature- and hash-verified, then installed per-user only when Python 3.8 x64 is absent |
| `windows/robomaster-0.1.1.68-cp38-cp38-win_amd64.whl` | `https://pypi.org/project/robomaster/0.1.1.68/` | `90A0EF0E5A95198FCDE0A37D5346E683A76814E0F61B0FAFFA9CD1FD7F109942` | Hash-verified and installed into `.venv-robot` |
| `windows/VisualCppRedist_AIO_20200707.exe` | DJI SDK commit `ff6646e115ab125af3207a4ed3df42cc76c795b2` | `F4D8654B0347827D2FEA21C5D1EAF94F471D4A4FA785CDD250E3CF736B035090` | Archive only; **unsigned and never executed automatically** |
| `windows/visualcppbuildtools_full.exe` | DJI SDK commit `ff6646e115ab125af3207a4ed3df42cc76c795b2` | `1E1774869ABD953D05D10372B7C08BFA0C76116F5C6DF1F3D031418CCDCD8F7B` | Archive only; never executed automatically |

The RoboMaster wheel contains DJI's CPython 3.8 `libmedia_codec.pyd` plus its
`avcodec-58.dll`, `avutil-56.dll`, `opus.dll`, `swresample-3.dll`, and
`swscale-5.dll` runtime files. DJI documents the SDK for EP/EP Core. A retail S1
still requires an independently validated compatible driver before physical motion.

The separately supplied
[Google Drive folder](https://drive.google.com/drive/folders/1AiKdzSjJ-HYY08YvImtp2t2a7SlUBsZl)
currently contains the RoboMaster PC desktop installer. That installer is downloaded
and verified by `setup_robomaster_pc.ps1`; it is too large for a normal GitHub file.

# Prospective local-header version clarification

Prepared before any VOST member request. The cached central-directory version 45 can reflect a ZIP64 local-header offset alone; that offset is not a field in the local header. Consequently, the local DEFLATE header may validly declare version 20. The transport now accepts local version 20 or 45 with the same required bit-3 flags, method, timestamp, literal name, CRC, size, bounded extras, DEFLATE termination, and next-header bounds. Local ZIP64 size fields still require version 45. The actual local version is recorded in each extracted member.

Reference: PKWARE APPNOTE 6.3.10, sections 4.4.3 and 4.5.3, [official format specification](https://pkware.cachefly.net/webdocs/casestudies/APPNOTE.TXT). This clarification does not claim to have read the archive's local headers. Earlier source snapshots and manufactured-test receipts remain unchanged.

The additional manufactured case has a central version 45 and an offset greater than 4 GiB, accepts local version 20 with ordinary zero/exact size fields, and rejects version 19 and version 20 with local ZIP64 size fields.

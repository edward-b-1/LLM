# List large files on `/`

```shell
find / -type f -printf "%s %p\n" 2>/dev/null | sort -nr | head -20 | awk '{printf "%10.2f MiB  %s\n", $1/1024/1024, $2}'
```
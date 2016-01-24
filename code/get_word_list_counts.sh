# author mittul
# date: 19.01.2016

tr ' ' '\n' |\
grep -v -P "^$" | \
sort | \
uniq -c | \
awk '{print $2" "$1}' | \
env LC_ALL=C sort -rnk2

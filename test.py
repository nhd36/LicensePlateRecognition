import re
str = u'[DMSM-8433] 加護亜依 Kago Ai – 加護亜依 vs. FRIDAY'.replace(" ", "_").replace("[].,", "")
regex = u'[\u3000-\u303f\u3040-\u309f\u30a0-\u30ff\uff00-\uff9f\u4e00-\u9faf\u3400-\u4dbf]+ (?=[A-Za-z ]+–)'
p = re.compile(regex, re.U)
match = p.sub("_", str)
#print(match.encode("UTF-8").decode("utf-8"))
print(match.encode("ascii"))
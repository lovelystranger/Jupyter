def nor(name):
    name = name.lower()
    return name[0].upper()+str(name[1:])
l1 = ['adam','LISA','barT']
l2 = map(nor,l1)
print(list(l2))

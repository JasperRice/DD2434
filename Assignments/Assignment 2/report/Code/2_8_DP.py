def tree_DP(root, k):
    # TODO: Implement algorithm for dynamic programming
    p = np.dot(root.cat, odd_DP(root, k))
    q = np.dot(root.cat, even_DP(root, k))
    print(p+q)
    return p

def odd_DP(node, k):
    if len(node.descendants) != 0:
        theta1 = np.array((node.descendants[0].cat))
        theta2 = np.array((node.descendants[1].cat))

        s11 = odd_DP(node.descendants[0], k)
        s21 = even_DP(node.descendants[1], k)
        s1  = np.dot(theta1, s11) * np.dot(theta2, s21)
        
        s12 = even_DP(node.descendants[0], k)
        s22 = odd_DP(node.descendants[1], k)
        s2  = np.dot(theta1, s12) * np.dot(theta2, s22)
        return s1 + s2

    s = np.zeros((k,1))
    s[1::2] = 1
    return s

def even_DP(node, k):
    if len(node.descendants) != 0:
        theta1 = np.array((node.descendants[0].cat))
        theta2 = np.array((node.descendants[1].cat))

        s11 = odd_DP(node.descendants[0], k) 
        s21 = odd_DP(node.descendants[1], k)
        s1  = np.dot(theta1, s11) * np.dot(theta2, s21)
        
        s12 = even_DP(node.descendants[0], k)
        s22 = even_DP(node.descendants[1], k)
        s2  = np.dot(theta1, s12) * np.dot(theta2, s22)
        return s1 + s2

    s = np.zeros((k,1))
    s[0::2] = 1
    return s
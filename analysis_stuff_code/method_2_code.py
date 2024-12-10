
def dfs(G, C, id, color):
    S = [id]
    C[id] = color

    while S:
        u = S.pop()
        for v in G[u]:
            if C[v] == -1:
                C[v] = color
                S.append(v)

if __name__ == '__main__':
    num_of_users, num_of_links = map(int, input().split(' '))
    links = [list(map(int, input().split(' '))) for _ in range(num_of_links)]
    num_of_queries = int(input())
    queries = [list(map(int, input().split(' '))) for _ in range(num_of_queries)]

    G = [[] for _ in range(num_of_users)]
    C = [-1] * num_of_users
    for f, t in links:
        G[f].append(t)
        G[t].append(f)

    color = 1
    for id in range(num_of_users):
        if C[id] == -1:
            dfs(G, C, id, color)
        color += 1

    for x, y in queries:
        if C[x] == C[y]:
            print('yes')
        else:
            print('no')

from collections import deque





def dfs(G, C, id, color):

    S = deque()

    S.append(id)

    C[id] = color



    while S:

        u = S.pop()

        #S = S[:-1]  # pop

        for i in range(len(G[u])):

            v = G[u][i]

            if C[v] == -1:

                C[v] = color

                S.append(v)





if __name__ == '__main__':

    # ??????????????\???

    num_of_users, num_of_links = [int(x) for x in input().split(' ')]

    links = []

    for _ in range(num_of_links):

        links.append(list(map(int, input().split(' '))))

    num_of_queries = int(eval(input()))

    queries = []

    for _ in range(num_of_queries):

        queries.append(list(map(int, input().split(' '))))



    # ???????????????

    G = [[] for _ in range(100000)]

    C = [-1] * 100000

    for f, t in links:

        G[f].append(t)

        G[t].append(f)



    color = 1

    for id in range(num_of_users):

        if C[id] == -1:

            dfs(G, C, id, color)

        color += 1



    # ???????????????

    for x, y in queries:

        if C[x] == C[y]:

            print('yes')

        else:

            print('no')
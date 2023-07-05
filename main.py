import random
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import itertools as it
from numba import njit
from numpy.linalg import matrix_power

#BASIC QUIVER RELATED FUNCTIONS
@njit
def mutate(A, i):
    span = [j for j in range(A.shape[0])]
    t = int(i) - 1
    span.remove(t)
    D = np.zeros(A.shape)
    D[t, :] = -A[t, :]
    D[:, t] = -A[:, t]
    for j in span:
        for k in span:
            D[j, k] = A[j, k] + np.sign(A[j, t] + A[t, k]) * np.maximum(0, A[j, t] * A[t, k])
    return D

@njit
def pos(A):
    M = np.zeros(A.shape)
    M[:,:] = np.maximum(A[:,:],0)
    return M

@njit
def ispath(A,start,end,length):
    M = pos(A)
    V = matrix_power(M,length)
    if V[start-1,end-1] != 0:
        return True
    else:
        return False
@njit
def iscycleno(A):
    r = 0
    lgth = [i for i in range(A.shape[0])]
    del lgth[0]
    del lgth[0]
    M = pos(A)
    for i in lgth:
        r += (np.trace(matrix_power(M,i)))
    return r

def list_to_mat(lst):
    k = len(lst[0])
    B = np.zeros([k,k])
    for i in range(len(lst)):
        for j in range(len(lst[i])):
            B[i,j+i] = lst[i][j]
    for k in range(1,len(lst[0])):
        for l in range(k,len(lst[0])):
            B[l,k] = - B[k,l]
    return B

def randy_bandy(B,l):
    while l > 0:
        i = random.randint(1,B.shape[0])
        B = mutate(B,i)
        l -= 1
    return B




def cyclist(A,t): #the slower but 'more accurate' cycle counting function
    i = t-1
    limit = A.shape[0]
    if limit < 3:
        return 0
    else:
        D = {}
        for j in range(3,min(5,limit)+1):
            D[j] = int(matrix_power(pos(A),j)[i,i])
        for k in range(6,limit+1):
            L = []
            s = k - 6
            for p in range(3,k):
                L.append(D[p])
            T = np.convolve(L,L)
            out = T[s]
            D[k] = int(matrix_power(pos(A),k)[i,i]) - int(out)
    return D




#Combinatorial functions for lists of mutations
def spacecreator(l, k):
    return (list(i) for i in it.permutations(l,k))

def invlist(l,k):
    M =[]
    for i in (it.product(l,repeat = k)):
        m = True
        p = 0
        while p < len(i)-1:
            if i[p] == i[p+1]:
                m = False
                break
            else:
                pass
            p+= 1
        if m:
            M.append(list(i))
    T = (r for r in M)
    return T
def modspacecreator(lst,k, r=3):
    t = k//r
    m = k - t*r
    L_1 = list(invlist(lst,r))
    D_1 = {}
    D_2 = {}
    for i in lst:
       D_1[i] = [l for l in L_1 if l[0] != i]
    R = (i for i in L_1)
    for _ in range(t-1):
       R = (r + n for r in R for n in D_1[r[-1]])
    if m ==0:
        return R
    else:
        L_2 = list(invlist(lst, m))
        for i in lst:
            D_2[i] = [l for l in L_2 if l[0] != i]
        R = (r + l for r in R for l in D_2[r[-1]])
        return R

def listsubtractor(L,M):
    for i in M:
        if i in L:
            L.remove(i)
    return L



#Class for a quiver defined via skew-symmetric matrix
class Quiver():
    def __init__(self,B,name):
        self.nodes = [i + 1 for i in range(B.shape[0])]
        self.B = B
        self.name = name


    def edges(self, A):
        Lst = []
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                if A[i, j] > 0:
                    Lst.append((i + 1, j + 1))
        return Lst

    def mut(self, A, LstMut):
        for j in LstMut:
            A = mutate(A, j)
        return A

    def path(self,A, L, cycle = False):
        m = True
        for i in range(len(L)-1):
            if A[L[i] - 1, L[i + 1] - 1] <= 0:
                m = False
                break
            else:
                pass
        if cycle:
            if A[L[-1]-1, L[0]-1] <= 0:
                m = False
        if m:
            return True
        else:
            return False

    def pathlist(self,k):
        lst = []
        sp= spacecreator(self.nodes,k)
        for i in sp:
            if self.path(self.B, i):
                lst.append(i)
        return lst

    def cycles(self,A, k):
        cyclelist = []
        lt = self.nodes.copy()
        for _ in range(k - 1):
            del lt[-1]
        for i in range(len(lt)):
            sp = spacecreator(self.nodes[i + 1:], k - 1)
            for r in sp:
                t = [lt[i]] + r
                if self.path(A,t,cycle=True):
                    cyclelist.append(t)
        cyclegen = (i for i in cyclelist)
        return cyclegen
    def cyclepairs(self, A):
        lst = []
        lgth = [i for i in range(len(self.nodes))] #needs A same shape as B
        del lgth[0]
        del lgth[0]
        for i in lgth:
            lst = it.chain(lst,self.cycles(A, i)) # was originally in list format
        p = []
        for i in lst:
            l = len(i)
            for j in range(len(i)):
                p.append((i[j % l], i[(j + 1) % l]))
        return p

    def cycleno(self,A):
        r = 0
        lgth = [i for i in range(len(self.nodes))]
        del lgth[0]
        del lgth[0]
        for i in lgth:
            r = r + sum(1 for _ in self.cycles(A,i))
        return r

    def loss_fn(self,A,lst):
        L = []
        for i in range(len(lst)):
            P = self.mut(A,lst[:i])
            L.append(iscycleno(P))
        return min(L)
    def loss_list(self, A,lst):
        L = []
        P = A
        L.append(iscycleno(A))
        for i in range(len(lst)):
            P = mutate(P, lst[i])
            L.append(iscycleno(P))
        return L

    def graph(self, Lst, save = False):
        G = nx.DiGraph()
        edges = self.edges(self.mut(self.B, Lst))
        cyc = self.cyclepairs(self.mut(self.B, Lst))
        G.add_nodes_from(self.nodes)
        G.add_edges_from(edges)
        em = listsubtractor(edges,cyc)
        for e in cyc:
            G[e[0]][e[1]]['color'] = 'red'
        for e in em:
            G[e[0]][e[1]]['color'] = 'black'
        edge_color_list = [G[e[0]][e[1]]['color'] for e in G.edges()]
        nx.draw(G, with_labels=True, edge_color = edge_color_list)
        if save:
            plt.savefig(f"{self.name}" + r".png")
        plt.show()

    def acyclic(self,A, k, r = 3, Graph = False):
        L = []
        D = {}
        Liist = list(modspacecreator(self.nodes,k,r = r))
        for (num, k) in enumerate(Liist):
            D[num-1] = k
        for i in Liist:
            L.append(self.loss_list(A,i))
        R = np.array(L)
        t = R.min()
        x_coord = np.where(R == t)[0]
        y_coord = np.where(R == t)[1]
        if t == 0:
            m = D[x_coord[0]]
            P = m[:y_coord[0]+1]
        else:
             q = []
             for j in x_coord:
                q.append(L[j][-1])
             index_min = min(range(len(q)), key=q.__getitem__)
             P = Liist[x_coord[index_min]]
        if Graph:
            M = self.mut(A,P)
            New = Quiver(M)
            New.graph([])
        return self.mut(A,P), P

    def shortest(self,A, k, r=3, Graph = False):
        L = []
        Liist = list(modspacecreator(self.nodes, k, r = r))
        for i in Liist:
            L.append(self.loss_fn(A, i)) # A has to be the same shape as B.
        index_min = min(range(len(L)), key=L.__getitem__) #thank you stack exchange
        Lst = Liist[index_min]
        L.clear()
        for j in range(len(Lst)):
           L.append(self.cycleno(self.mut(A,Lst[:j])))
        ind_min = min(range(len(L)), key=L.__getitem__)
        P = Lst[:ind_min]
        if Graph:
            M = self.mut(A,P)
            New = Quiver(M,'')
            New.graph([])
        return self.mut(A,P), P

    def acyclicoptimiser(self,k, r = 3, Graph = False, verbose = False, save = False):
        Mat = [self.B]
        Mut = []
        C = self.B
        while iscycleno(self.mut(self.B,Mut)) != 0:
            C, P = self.acyclic(C,k, r = r)
            Mut = Mut + P
            for j in range(len(Mat)):
                if np.array_equal(Mat[j],C):
                    i = random.randint(1,self.B.shape[0])
                    C = mutate(C,i)
            Mat.append(C)
        if Graph:
            M = self.mut(self.B, Mut)
            New = Quiver(M, name = self.name)
            New.graph([], save = save)
        if verbose:
            return self.mut(self.B, Mut), Mut
        else:
            return self.mut(self.B, Mut)
    def findmutacyc(self,k):
        Mat = [self.B]
        Mut = []
        C = self.B
        while iscycleno(self.mut(self.B,Mut)) != 0:
            C, P = self.shortest(C,k)
            Mut = Mut + P
            for j in range(len(Mat)):
                if np.array_equal(Mat[j],C):
                    i = random.randint(1,self.B.shape[0])
                    C = mutate(C,i)
            Mat.append(C)
        return Mut
    def eulerform(self,a,b):
        counter = 0
        for i in range(len(a)):
            counter += a[i]*b[i]
        for i in range(len(a)):
            for j in range(len(b)):
                if pos(self.B)[i,j] != 0:
                    counter -= a[i]*b[j]
        return counter


def instate(k, n):
    A = np.zeros([(k - 1) * (n - k - 1), (k - 1) * (n - k - 1)])
    for i in range(n - k - 2):
        for j in range(k - 1):
            t = j * (n - k - 1) + i
            A[t, t + 1] = 1
            A[t + 1, t] = -1  # deals with the horizontal square maps

    for i in range(n - k - 1):
        for j in range(k - 2):
            t = j * (n - k - 1) + i
            A[t, t + (n - k - 1)] = 1
            A[t + (n - k - 1), t] = -1  # deals with vert maps

    for i in range(n - k - 2):
        for j in range(k - 2):
            t = j * (n - k - 1) + i
            A[t, t + (n - k)] = -1
            A[t + (n - k), t] = 1
    return A






class Gr(Quiver):
    def __init__(self, k, n, name):
        self.k = k
        self.n = n
        self.instate = instate(self.k, self.n)
        self.nodes = [i + 1 for i in range((self.k - 1) * (self.n - self.k - 1))]
        self.edge = self.edges(self.instate)
        self.name = name
        super().__init__(self.instate, self.name)

    def loc(self, i):
        j = (i - 1) % (self.n - self.k - 1) + 1
        l = np.floor(float(i - 1) / (float(self.n - self.k - 1)))
        return (l, j)

    def tiltgraph(self, Lst):
        G = nx.DiGraph()
        cyc= self.cyclepairs(self.mut(self.instate, Lst))
        edges = self.edges(self.mut(self.instate, Lst))
        for i in self.nodes:
            G.add_node(i, pos=self.loc(i))
        G.add_edges_from(edges)
        em = listsubtractor(edges, cyc)
        for e in cyc:
            G[e[0]][e[1]]['color'] = 'red'
        for e in em:
            G[e[0]][e[1]]['color'] = 'black'
        edge_color_list = [G[e[0]][e[1]]['color'] for e in G.edges()]
        pos = nx.get_node_attributes(G, 'pos')
        nx.draw(G, pos, with_labels=True, edge_color=edge_color_list)
        plt.show()


def acycletester(l,p):
    B = np.zeros([p,p])
    for i in range(p-l-1):
        B[i,i+1] = 1
        B[i+1,i] = -1
    B[p-l-1,0] = 1
    B[0,p-l-1] = -1
    for j in range(l):
        B[p-l+j,p-2*l +j] = 1
        B[p-2*l+j,p-l+j] = -1
    return B


class Gr_kappa:
    def __init__(self,k,n):
        self.k = k
        self.n = n
    def enhance(self, L):
        B = [l+1 for l in L]
        New = [self.n - self.k] + B
        return New
    def comp(self, L1,L2):
        l = len(L1)
        L = L2[:l]
        Com = [a < b for (a,b) in zip(L1,L)]
        return Com
    def indToPart(self, L):
        L.sort(reverse=True)
        P = [i+1 for i in range(len(L))]
        P.sort(reverse=True)
        T = [l-p for (l,p) in zip(L,P)]
        return T
    def kappa(self,J,I): #this works
        mu = self.indToPart(I)
        nu = self.indToPart(J)
        if any(self.comp(mu,nu)):
            counter = 1
            part = self.enhance(mu)
            while any(self.comp(part,nu)):
                part = self.enhance(part)
                counter += 1
            return counter
        else:
            return 0
    def lam(self,l1,l2):
        return self.kappa(l2,l1)-self.kappa(l1,l2)
    def lam_list(self,a,b,c,d):
        counter = 0
        for i in range(len(a)):
            for j in range(len(b)):
                counter += c[i]*d[j]*self.lam(a[i],b[j])
        return counter

    def dec(self,list,l2):
        counter = 0
        while len(list) > 1:
            for i in range(1,len(list)):
                counter += l2[0]*l2[i]*self.lam(list[i],list[0])
            del list[0]
            del l2[0]
        return counter/2
    def left_to_right(self,l1,l2):
        counter = 0
        while len(l1) > 0:
            for j in range(len(l2)):
                counter += self.lam(l1[-1],l2[j])
            del l1[-1]
        return counter
    def inv_left_to_right(self,l1,l2):
        counter = 0
        while len(l1) > 0:
            for j in range(len(l2)):
                counter += (-1)*self.lam(l1[-1],l2[j])
            del l1[-1]
        return counter
    def lam_matrix(self,l):
        k = len(l)
        B = np.zeros([k,k])
        for i in range(k):
            for j in range(k):
                B[i,j] = self.lam(l[i],l[j])
        return B
    def ind_swap(self,l1,l2,l3,l4):
        return self.dec(l1,l2) - self.dec(l3,l4)


def edge(A):
    Lst = []
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if A[i, j] > 0:
                Lst.append((i + 1, j + 1))
    return Lst

def tilttester(k,n):
    B = np.zeros([n,n])
    B[n-1,k-1] = 1
    B[k-1,n-1] = -1
    for i in range(k):
        B[i,(i+1)] = -1
        B[(i+1),i] = 1
    for j in range(k,n-2):
        B[j,(j+1)] = 1
        B[(j+1),j] = -1
    L = [i for i in range(k,n)]
    for i in L:
        B = mutate(B,i)
    G = nx.DiGraph()
    G.add_nodes_from([i + 1 for i in range(B.shape[0])])
    G.add_edges_from(edge(B))
    nx.draw(G)
    plt.show()

# Q = Gr_kappa(3,6)
# l = [[1,2,3],[2,3,4],[3,4,5],[4,5,6],[1,5,6],[1,2,6],[1,2,4],[1,2,5],[1,3,4],[1,4,5]]
# t = [[1,2,3],[2,3,4],[3,4,5],[4,5,6],[1,5,6],[1,2,6],[1,3,5],[1,2,5],[1,3,4],[1,4,5]]
# A = Q.lam_matrix(l)
# # # B = Q.lam_matrix(t)
# ind_235 = np.array([[0,1,0,0,0,0,-1,1,0,0]])
# # w_3 = np.array([[0,0,0,0,0,0,-1,1,1,0]])
# ind_356 = np.array([[0],[0],[1],[0],[1],[0],[0],[0],[0],[-1]])
#
# ind_m1 = ind_235 + ind_356.transpose()
#
#
#
# ind_246 = np.array([[0],[1],[0],[0],[0],[1],[0],[-1],[-1],[1]])
#
# b_1 = np.array([[-1,0,0,0,0,0,0,1,1,-1]])
# b_4 = np.array([[0,0,0,0,-1,1,-1,0,0,1]])
# b_2 = np.array([[0,1,-1,0,0,0,-1,0,0,1]])
# b_3 = np.array([[0,0,1,-1,1,0,1,-1,-1,0]])
#
#
#
# print(0.5*np.matmul(np.matmul(ind_m1 - b_3,A),ind_246 - b_2.transpose() - b_4.transpose()))





# m_1 = [[1,2,3],[1,4,5]]
# m_2 = [[1,2,5],[1,3,4]]
# m_3 = [[1,2,4]]
# m_4 = [[1,2,6],[2,3,4]]
# m_5 = [[1,2,5],[1,3,4]]
# t_1 = 0
# t_2 = 0
# for i in m_1:
#     for j in m_4:
#         t_1 += Q.lam(i,j)
# for i in m_2:
#     for j in m_4:
#         t_2 += Q.lam(i,j)
# for i in m_1:
#     for j in m_5:
#         t_1 += Q.lam(j,i)
# for i in m_2:
#     for j in m_5:
#         t_2 += Q.lam(j,i)
# for i in m_3:
#     for j in m_4:
#         t_1 += Q.lam(j,i)
#         t_2 += Q.lam(j,i)
# for i in m_3:
#     for j in m_5:
#         t_1 += Q.lam(i,j)
#         t_2 += Q.lam(i,j)
#
# print(0.5*t_1)
# print(0.5*t_2)
#
#
# Q = Gr_kappa(2,6)
#
#
# a_1 = [[1,3],[1,2],[3,5]]
# s_1 = [-1,1,1]
#
# a_2 = [[1,3],[1,5],[2,3]]
# s_2 = [-1,1,1]
#
# a_3 = [[3,5],[2,3],[4,5]]
# s_3 = [-1,1,1]
#
# a_4 = [[1,3],[3,5],[1,5],[2,3],[3,4]]
# s_4 = [-1,-1,1,1,1]
#
# a_5 = [[1,3],[1,2],[3,4]]
# s_5 = [-1,1,1]
#
# a_6 = [[1,3],[1,6],[2,3]]
# s_6 = [-1,1,1]
#
# a_7 = [[1,3],[1,5],[1,2],[1,6],[3,5]]
# s_7 = [-1,-1,1,1,1]
#
# a_8 = [[1,5],[1,2],[5,6]]
# s_8 = [-1,1,1]
#
#
# l_1 = [a_1,a_2]
# sign_1 = [s_1,s_2]
#
# l_2 = [a_3,a_4,a_5]
# sign_2 = [s_3,s_4,s_5]
#
# l_3 = [a_6,a_7,a_8]
# sign_3 = [s_6,s_7,s_8]
#
# for k in range(len(l_1)):
#     for j in range(len(l_2)):
#         for i in range(len(l_3)):
#             print(l_1[k] + l_2[j] + l_3[i])


Q = Gr_kappa(2,8)
l = [[1,2],[2,3],[3,4],[4,5],[5,6],[6,7],[7,8],[1,8],[1,3],[1,5],[1,7],[3,5],[5,7]]
A = Q.lam_matrix(l)
ind_24 = np.array([[0,1,0,1,0,0,0,0,0,0,0,-1,0]])
ind_26 = np.array([[0],[1],[0],[0],[0],[1],[0],[0],[-1],[1],[0],[0],[-1]])
ind_28 = np.array([[0],[1],[0],[0],[0],[0],[0],[1],[-1],[0],[0],[0],[0]])
ind_46 = np.array([[0],[0],[0],[1],[0],[1],[0],[0],[0],[0],[0],[0],[-1]])
ind_68 = np.array([[0],[0],[0],[0],[0],[1],[0],[1],[0],[0],[-1],[0],[0]])
b_1 = np.array([[-1,1,0,0,0,0,0,0,0,1,0,-1,0]])
b_2 = np.array([[0,0,-1,1,0,0,0,0,1,-1,0,0,0]])
b_3 = np.array([[0,0,0,0,0,0,0,0,-1,0,1,1,-1]])
b_4 = np.array([[0,0,0,0,-1,1,0,0,0,1,-1,0,0]])
b_5 = np.array([[0,0,0,0,0,0,-1,1,0,-1,0,0,1]])


def ind_check(ind,l):
    if ind.shape[0] == 1:
        return [(p,sorted(t)) for (t,p) in zip(l,ind.tolist()[0]) if p != 0]
    else:
        k = [i for [i] in ind.tolist()]
        return [(p,sorted(t)) for (t,p) in zip(l,k) if p != 0]


m_1 = ind_24 + ind_26.transpose()
m_2 = ind_24 + ind_26.transpose() + ind_28.transpose()
m_3 = ind_24 + ind_26.transpose() + ind_28.transpose() + ind_46.transpose()

# print(0.5*np.matmul(np.matmul(ind_24,A),ind_26))

# q_1 = [[],[2],[1,2]]
# q_2 = [[],[1],[4],[1,4],[1,3,4]]
# q_3 = [[],[1],[1,3],[1,3,5]]
# q_4 = [[],[4],[3,4],[2,3,4]]
# q_5 = [[],[5],[4,5]]
#
#
# comp = []
# for i_1 in range(len(q_1)):
#     for i_2 in range(len(q_2)):
#         for i_3 in range(len(q_3)):
#             for i_4 in range(len(q_4)):
#                 for i_5 in range(len(q_5)):
#                     if sorted(q_1[i_1] + q_2[i_2] + q_3[i_3] + q_4[i_4] + q_5[i_5]) == [1,2,3,4,5]:
#                         comp.append([q_1[i_1],q_2[i_2],q_3[i_3],q_4[i_4],q_5[i_5]])
# print(len(comp))
# for i in range(len(comp)):
#     print(comp[i])   #very useful trick when computing single coefficient!!










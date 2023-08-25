#!/usr/bin/env python
# coding: utf-8

# In[1]:


#pip install numpy


# In[2]:


import numpy as np


# In[3]:


arr1 = np.arange(1000)


# In[4]:


get_ipython().run_line_magic('time', 'arr2 = arr1 * 2')


# In[5]:


lst1 = list(range(1000))


# In[6]:


get_ipython().run_line_magic('time', 'lst2 = [i * 2 for i in lst1]')


# In[7]:


x = np.array([[1, 2] , [3, 4]])


# In[8]:


x


# In[9]:


x.ndim


# In[10]:


x.shape


# In[11]:


x.dtype


# In[12]:


x * 3


# In[13]:


x + x


# In[14]:


x * x


# In[15]:


1 / x


# In[16]:


x ** 5


# In[17]:


y = np.array([[-2, 6],[8, 3]])


# In[18]:


y


# In[19]:


x


# In[20]:


x < y


# In[21]:


lst = [2, 4.5 , 6]
type(lst)


# In[22]:


arr = np.array(lst)
arr


# In[23]:


arr.dtype


# In[24]:


lst2 = [[1, 2, 3] , [4, 5, 6]] 
arr2 = np.array(lst2)
arr2


# In[25]:


arr2.shape


# In[26]:


arr2.ndim


# In[27]:


arr  = np.zeros(3)
arr


# In[28]:


arr = np.full(3, 2)
arr


# In[29]:


x = np.full((3, 2), 4)
x


# In[30]:


a = np.identity(3)
a


# In[31]:


lst = [1, 2]
arr = np.array(lst, dtype=np.int32)
arr.dtype


# In[32]:


arr2 = np.array(lst, dtype=np.float64)
arr2.dtype


# In[33]:


arr2


# In[34]:


x = np.array(lst)
x.dtype


# In[35]:


y = x.astype(np.float64)
y.dtype


# In[36]:


lst2 = [1.6, 3.2, 0.3]
a = np.array(lst2)
a


# In[37]:


a.astype(np.int32)


# indexing and slicing

# In[38]:


arr = np.arange(8)
arr


# In[39]:


arr[3]


# In[40]:


arr[2:5]


# In[41]:


arr[2:5] = 13
arr


# In[42]:


x = arr[2:6]
x


# In[43]:


x[1] = 17
x


# In[44]:


arr


# In[45]:


x[:] = 64
x


# In[46]:


arr


# In[47]:


arr = np.arange(8)
arr


# In[48]:


arr[2:6].copy()


# In[49]:


a = np.array([[1, 2, 3] , [4, 5, 6], [7, 8, 9]])
a


# In[50]:


a[2]


# In[51]:


a[0]


# In[52]:


a[0][2]


# In[53]:


a[2][1]


# In[54]:


a[2][2]


# In[55]:


a[2, 2]


# In[56]:


a.ndim


# # arr3d

# In[57]:


a = np.array([[[1, 2, 3] , [4, 5, 6]],[[7, 8, 9],[10, 11, 12]]])
a


# In[58]:


a.ndim


# In[59]:


a.shape


# In[60]:


a[0]


# In[61]:


a[1]


# In[62]:


a[0][0]


# In[63]:


a[0][1]


# In[64]:


a[1][0]


# In[65]:


a[1][1]


# In[66]:


a[1][1][1]


# In[67]:


a[1][1][0]


# In[68]:


a[0]


# In[69]:


y = a[0].copy()
y


# In[70]:


a[0] = 88
a


# In[71]:


y


# In[72]:


a[0] = y
a


# # arr2d

# In[73]:


x = np.array([[1, 2, 3],[4, 5, 6],[7, 8, 9]])
x


# In[74]:


x.shape


# In[75]:


x.ndim


# In[76]:


x[:2]


# In[77]:


x[:1]


# In[78]:


x[:]


# In[79]:


x[:2, 1:]


# In[80]:


x[:2, 1:] = 0
x


# In[81]:


x = np.array([[1, 2, 3],[4, 5, 6],[7, 8, 9]])
x


# In[82]:


x[:2, 0]


# In[83]:


x[:1, 1]


# In[84]:


x[:, :1]


# In[85]:


x[:, :2]


# # Boolean indexing

# In[86]:


n = np.array(['ali', 'sara', 'taha', 'ali'])
n


# In[87]:


n == 'ali'


# In[88]:


d = np.random.randn(4,3)
d


# In[89]:


d [n == 'ali']


# In[90]:


d[n=='ali' , 1:]


# In[91]:


d


# In[92]:


d [~(n == 'ali')]


# In[93]:


c = n == 'ali'
d[~c]


# In[94]:


d


# In[95]:


m = (n == 'ali') | (n == 'taha')
d[m]


# In[96]:


x = np.random.randn(3,4)
x


# In[97]:


x[x < 0]  = 0
x


# # Fancy indexing: indexing using integer arrays.

# In[98]:


arr = np.empty((7,5))
for i in range(7):
    arr[i] = 5*i+1
arr    


# In[99]:


arr[[3, 6, 0]]


# In[100]:


arr[[-7, -1]]


# In[101]:


x = np.arange(35).reshape((7,5))
x


# In[102]:


x[[1, 5, 6, 2] , [0, 3, 1, 2]]


# In[103]:


x[[2, 6]][:,[0, 3 , 1]]


# In[104]:


# Transposing arrays and swapping axes


# In[105]:


arr = np.arange(8).reshape((2,4))
arr


# In[106]:


arr.T


# In[107]:


z = np.arange(60).reshape((3, 4, 5))


# In[108]:


z


# In[109]:


z.swapaxes(0, 1)


# In[110]:


z.transpose((1, 0, 2))


# # Universal Function : ufunc

# In[111]:


arr = np.arange(4)
arr


# In[112]:


np.sqrt(arr)


# In[113]:


np.exp(arr)


# In[114]:


x = [2.6, 8.5, -9]
r, w = np.modf(x)


# In[115]:


r


# In[116]:


w


# In[117]:


x = np.random.randn(4)
y = np.random.randn(4)


# In[118]:


x


# In[119]:


y


# In[120]:


np.maximum(x, y)


# # where

# In[121]:


arr1 = np.array([1, 5, 8])
arr2 = np.array([4, 7, 12])
cond = np.array([True,False,True])


# In[122]:


r = [(x if c else y)
    for x, y, c in zip(arr1, arr2, cond)]


# In[123]:


r


# In[124]:


res = np.where(cond, arr1 , arr2)
res


# In[125]:


x = np.random.randn(2, 3)
x


# In[126]:


x > 0


# In[127]:


np.where(x > 0 , 1 , 0)


# In[128]:


x


# In[129]:


np.where(x > 0 , 1 , x)


# In[130]:


score = np.array([[7,12,20],[10,15,4]])
score


# In[131]:


np.where(score>10 , score , 10)


# In[132]:


x = [[1,2],[3,4]]
y = [[5,6],[7,8]]
c = [[True,False],[False,True]]
np.where(c, x, y)


# ## Mathematical and Statistical Methods

# In[133]:


# amin, amax , mean , averge , nanmean


# In[134]:


arr = np.array([4,2,3,1])
np.amin(arr)


# In[135]:


np.amax(arr)


# In[136]:


np.mean(arr)


# In[137]:


np.average(arr,weights=[1,2,3,4])   # (4*1 + 2*2 + 3*3 + 1*4 )/ 10


# In[138]:


np.nan


# In[139]:


x = np.array([1, 4, np.nan, 8, np.nan, 7])
np.mean(x)


# In[140]:


np.nanmean(x)   # 20/4


# ## var , std , median , quantile , percentile

# In[141]:


x = np.array([3, 5, 9, 8, 1, 4, 12, 17, 6])


# In[142]:


np.var(x)


# In[143]:


np.std(x)


# In[144]:


np.median(x)


# In[145]:


np.sort(x)


# In[146]:


y = np.array([3, 5, 9, 8, 1, 4, 12, 17])
np.median(y)


# In[147]:


np.sort(y)


# In[148]:


a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
np.quantile(a, 0.25)


# In[149]:


np.quantile(a,0.50)


# In[150]:


np.quantile(a,0.75)


# In[151]:


np.percentile(a,25)


# In[152]:


np.percentile(a,50)


# In[153]:


np.percentile(a,75)


# In[154]:


np.percentile(a,10)


# In[155]:


# sum , cumsum


# In[156]:


arr = np.array([1, 2, 3, 4])


# In[157]:


np.sum(arr)


# In[158]:


np.cumsum(arr)


# In[159]:


x = np.array([[1,2,3],[4,5,6],[7,8,9]])
x


# In[160]:


np.sum(x)


# In[161]:


np.sum(x, axis=0)


# In[162]:


np.sum(x, axis=1)


# In[163]:


x


# In[164]:


np.cumsum(x, axis=0)


# In[165]:


np.cumsum(x, axis=1)


# In[166]:


# all , any


# In[167]:


a = np.array([True, True, False])


# In[168]:


a.any()


# In[169]:


a.all()


# In[170]:


b = [0, 2, -3]


# In[171]:


np.any(b)


# In[172]:


np.all(b)


# In[173]:


c = [8, 2, -3]


# In[174]:


np.all(c)


# In[175]:


# unique


# In[176]:


arr = np.array([3, 4, 7, 4, 2, 1, 3, 5, 4, 4])
np.unique(arr)


# In[177]:


a , i = np.unique(arr, return_index=True)


# In[178]:


a


# In[179]:


i


# In[180]:


# sort


# In[181]:


data = [('ali',12.5,35) , ('sara',18.75,27),('taha',16.25,27)]
type(data)


# In[182]:


d = [('name','S10'), ('score',float) , ('age',int)]


# In[183]:


arr = np.array(data, dtype=d)


# In[184]:


np.sort(arr, order='age')


# In[185]:


np.sort(arr, order='score')


# In[186]:


#in1d


# In[187]:


x = np.array([7, 1, 4, 2, 5, 7])
y = [3, 4, 7]
np.in1d(x, y)


# In[188]:


x = np.array([7, 1, 4, 2, 5, 7])
y = [3, 4, 7]
m = np.in1d(x, y)
x[m]


# In[189]:


x = np.array([7, 1, 4, 2, 5, 7])
y = [3, 4, 7]
m = np.in1d(x, y, invert=True)
x[m]


# In[190]:


# save , load


# In[191]:


a = np.array([4, 6, 9])


# In[192]:


np.save('test.npy', a)


# In[193]:


np.load('test.npy')


# In[194]:


with open('test.npy', 'wb') as f:
    np.save(f, a)


# In[195]:


with open('test.npy', 'rb') as f:
    x = np.load(f)


# In[196]:


x


# In[197]:


arr1 = np.array([1, 2])
arr2 = np.array([3, 4, 5])
np.savez('test2.npz',x=arr1, y=arr2)


# In[198]:


t = np.load('test2.npz')
t['x']


# In[199]:


t['y']


# In[200]:


t.files


# ### random_sample  U[a,b) : (b-a)*random_sample(...) + a

# In[201]:


# U[1,20)
19 * np.random.random_sample(5) + 1


# ### rand    : U[0,1)

# In[203]:


np.random.rand(3)


# ### randint 

# In[208]:


np.random.randint(0,11,7)


# ### randn : N(mu,sigma^2) : sigma * randn(...) + mu

# In[209]:


# N(3, 2.5^2)
2.5 * np.random.randn(4) + 3


# In[218]:


mu = 5
sigma = 0.8
s = np.random.normal(mu, sigma, 5000)


# In[219]:


import matplotlib.pyplot as plt
plt.hist(s, 30, density=True)
plt.show()


# In[222]:


a = np.random.normal(size=3)
a


# In[225]:


np.random.seed(1234)
b = np.random.normal(size=3)
b


# In[ ]:


## inner , outer


# In[226]:


a = np.array([1, 2, 3])
b = np.array([5, 6, 0])


# In[227]:


np.inner(a, b)


# In[228]:


1*5 + 2*6 + 3*0


# In[229]:


np.outer(a, b)


# In[231]:


x = np.array([[1, 2],
              [3, 4]])

y = np.array([[5, 6],
              [7, 8]])


# In[233]:


np.dot(x, y)


# In[234]:


from numpy.linalg import inv


# In[235]:


inv(x)


# In[236]:


from numpy.linalg import qr


# In[237]:


b = x.T.dot(x)


# In[238]:


i = inv(b)


# In[239]:


b.dot(i)


# In[240]:


q, r= qr(b)


# In[241]:


q


# In[242]:


r


# In[ ]:





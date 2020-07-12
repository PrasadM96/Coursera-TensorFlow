#!/usr/bin/env python
# coding: utf-8

# In[23]:


import tensorflow as tf
import numpy as np
from tensorflow import keras

model= tf.keras.Sequential([keras.layers.Dense(units=1,input_shape=[1])])
model.compile(optimizer='sgd',loss='mean_squared_error')

xs=np.array([-1.0,0.0,1.0,2.0,3.0,4.0], dtype=float)
ys=np.array([-3.0,-1.0,1.0,3.0,5.0,7.0],dtype=float)

model.fit(xs,ys,epochs=2000)



# In[25]:


print(model.predict([10.0]))


# In[ ]:


%%javascript
<!-- Save the notebook -->
IPython.notebook.save_checkpoint();

%%javascript
IPython.notebook.session.delete();
window.onbeforeunload = null
setTimeout(function() { window.close(); }, 1000);
#### Build PyTorch Extensions

**NOTE:** PyTorch >= 1.4, CUDA >= 9.0 and GCC >= 4.9 are required.

```
GRNET_HOME=`pwd`

# Chamfer Distance
cd $GRNET_HOME/extensions/chamfer_dist
python setup.py install --user

# Cubic Feature Sampling
cd $GRNET_HOME/extensions/cubic_feature_sampling
python setup.py install --user

# Gridding & Gridding Reverse
cd $GRNET_HOME/extensions/gridding
python setup.py install --user

# Gridding Loss
cd $GRNET_HOME/extensions/gridding_loss
python setup.py install --user
```

#### Dataset Generation Steps

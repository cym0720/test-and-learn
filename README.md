# 陈一鸣能量机关作业
## 完成情况
  真的真的真的就差拟合，卡尔曼滤波器都写好了，但是速度拟合部分进度过慢，目前只掌握了梯
  度下降的理论部分，代码部分还没有系统实践
## 代码&算法说明
###算法说明
  识别部分逻辑十分简单但是高效，参照[桂林理工大学开源](https://github.com/DH13768095744/RM_Buff_Tracker_GUT/blob/main/readme.md)的绘制圆形蒙版遮罩思路，以实心圆遮住能量机关中心“R”到靶子的部分，并且绘制大号空心圆套住整个能量机关使得靶子被提取出来，
  之后用旋转矩形与二值化后的面积差直接识别。
  拟合部分考虑用梯度下降（有好用的请学长学姐推荐一下孩子要哭了）
###代码
  里面有很多函数，但是只看recognition应该就行了，其余都是铺垫，识别后的视频输出到了build文件夹的output.mp4里面

  
  

# Write good commit message



[type] short subject (1 line) to state this commit: fix bug xxx, add feature: xxxx

A story explaining what is the problem and how this commit fix it.

**Type：类型**

具体来说，Type 分为：

- **feat:** 增加新功能；
- **fix:** 修复错误；
- **docs:** 修改文档；
- **style:** 修改样式；
- **refactor:** 代码重构；
- **test:** 增加测试模块，不涉及生产环境的代码；
- **chore:** 更新核心模块，包配置文件，不涉及生产环境的代码；

不要用 git commit -m 改用 git commit -v 即可。

每次一个小的改动就是一个commit,不要一大堆改动一个commit,不然review的时候痛苦到死.Minimal modification each commit.




## Reference:

[写出好的 commit message · Ruby China (ruby-china.org)](https://ruby-china.org/topics/15737)
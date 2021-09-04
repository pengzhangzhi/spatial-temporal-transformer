[TOC]



# Tools for Writing:

## Editor:

[Your Projects - Overleaf, Online LaTeX Editor](https://www.overleaf.com/project)

## translation:

[DeepL翻译：全世界最准确的翻译](https://www.deepl.com/translator)

Google Translation

## latex:

Mathpix Snipping Tool

[在线LaTeX公式编辑器-编辑器 (latexlive.com)](https://www.latexlive.com/##)

[Detexify LaTeX handwritten symbol recognition (kirelabs.org)](http://detexify.kirelabs.org/classify.html)

[CTAN: /tex-archive/support/excel2latex](https://ctan.org/tex-archive/support/excel2latex)

## wording:

Usage of word: [Linggle - Language Reference Search Engines - NLPLab](https://linggle.com/)

## citation:

Cite papers: [Semantic Scholar | AI-Powered Research Tool](https://www.semanticscholar.org/)





# Architecture:

![image-20210720222421053](Latex%20cheatsheet.assets/image-20210720222421053.png)

## Chapters folder :

each .tex file contains specific content of each chapers. 

![image-20210720222455943](Latex%20cheatsheet.assets/image-20210720222455943.png)





## Images: 

contains all the figures in your paper.

![image-20210720222555582](Latex%20cheatsheet.assets/image-20210720222555582.png)

## main.tex：

Set up paper achitecture, highly recommend that using template to construct you paper.  And import content in chapter folder into main.tex.

```latex

\section{Related Work}
\input{Chapters/related work}

\section{METHODOLOGY}
\input{Chapters/method}

\section{Experiment Setup}
\input{Chapters/experiment setup}

\section{Comparison and Analysis of Results}
\input{Chapters/result}

\section{ablation study}
\input{Chapters/ablation}

\section{hyper-parameter study}
\input{Chapters/hyper-parameter study}

\section{Conclusion}
\input{Chapters/conclusion}
```



# Math：

## tools:

## basis:

## examples:

# Table：

## Figure:

## basis:

## examples:

# Bibliography:

**Use BibTex:**

1. New a .bib file for manage Bibliography. For example, `bibliography.bib`.

   ![image-20210722085210317](Latex%20cheatsheet.assets/image-20210722085210317.png)

2.  save bib format citaion into it.

   

   ![image-20210722085231057](Latex%20cheatsheet.assets/image-20210722085231057.png)

   

```text
@article{He2016DeepRL,
  title={Deep Residual Learning for Image Recognition},
  author={Kaiming He and X. Zhang and Shaoqing Ren and Jian Sun},
  journal={2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2016},
  pages={770-778}
}
```

`He2016DeepRL` is the name to cite in your latex file.

3. Insert the following code into the end of your paper(but before `\end{document}`)：

```latex
\bibliographystyle{IEEEtran}
\bibliography{bibliography.bib}
```

These code would create a Reference section in your paper.

![image-20210722085152536](Latex%20cheatsheet.assets/image-20210722085152536.png)

`bibliographystyle` indicates the style of refences, which varies with your magazine.

> IEEEtran表示的是调用模板自带的格式，一般是IEEEtran.cls文件定义的，IEEEexample就是你制作好的bibtex文件。

- plain，按字母的顺序排列，比较次序为作者、年度和标题；
- unsrt，样式同plain，只是按照引用的先后排序；
- alpha，用作者名首字母+年份后两位作标号，以字母顺序排序；
- abbrv，类似plain，将月份全拼改为缩写，更显紧凑；
- ieeetr，国际电气电子工程师协会期刊样式；
- acm，美国计算机学会期刊样式；
- siam，美国工业和应用数学学会期刊样式；
- apalike，美国心理学学会期刊样式；

4. To cite a paper:

   ```latex
   \cite{He2016DeepRL}
   ```

5. Make your citaion more fancy:

   ```latex
   \usepackage[backref]{hyperref}
   ```

Detailed tutorial：

[Latex参考文献管理：Bibtex教程 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/114733612)

[在LaTeX中如何引用参考文献 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/265479955)


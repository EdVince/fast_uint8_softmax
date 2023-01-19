# fast uint8 softmax

speed for size 256*256*256:

| Softmax | Softmax-base2 | Softmax-base2-fast |
| ------- | ------------- | ------------------ |
| 115717us | 108714us     | 23716us            |

## Process

$p_i=\frac{e^{x_i}}{\sum e^{x_k}} \approx \frac{2^{x_i}}{\sum 2^{x_k}}$

Setting the input $x_i$ is uint8 and in the range (0,127).

1. in "float" define: $a = 2^{exp} * (1 + mant)$
2. let $a_i = 2^{x_i}$, where $exp = x_i$ and $mant = 0$
3. so $\sum 2^{x_k} = sum = 2^{exp_{sum}}*(1+mant_{sum})$
4. put them back: $p_i=\frac{2^{x_i}}{2^{exp_{sum}}*(1+mant_{sum})}=\frac{2^{x_i}}{2^{exp_{sum}}}\frac{1}{1+mant_{sum}}$
5. reverse the mant: $p_i=2^{x_i-exp_{sum}-1}*\frac{2}{1+mant_{sum}}$
6. finally: $p_i=2^{exp}*(1+mant)$, where $exp = x_i-exp_{sum}-1$ and $(1+mant) = \frac{2}{1+mant_{sum}}$


## Reference
1. [paper](https://www.nature.com/articles/s41598-021-94691-7)
2. [FloatConverter](https://www.h-schmidt.net/FloatConverter/IEEE754.html)

# Mammoth - An Extendible (General) Continual Learning Framework for Pytorch

<p align="center">
  <img width="112" height="112" src="seq_mnist.gif" alt="Sequential MNIST">
  <img width="112" height="112" src="seq_cifar10.gif" alt="Sequential CIFAR-10">
  <img width="112" height="112" src="seq_tinyimg.gif" alt="Sequential TinyImagenet">
  <img width="112" height="112" src="perm_mnist.gif" alt="Permuted MNIST">
  <img width="112" height="112" src="rot_mnist.gif" alt="Rotated MNIST">
  <img width="112" height="112" src="mnist360.gif" alt="MNIST-360">
</p>

## Setup

+ Use `./utils/main.py` to run experiments.
+ Use argument `--load_best_args` to use the best hyperparameters from the paper.
+ New models can be added to the `models/` folder.
+ New datasets can be added to the `datasets/` folder.

## Models

+ Gradient Episodic Memory (GEM)
+ A-GEM
+ A-GEM with Reservoir (A-GEM-R)
+ Experience Replay (ER)
+ Meta-Experience Replay (MER)
+ Function Distance Regularization (FDR)
+ Greedy gradient-based Sample Selection (GSS)
+ Hindsight Anchor Learning (HAL)
+ Incremental Classifier and Representation Learning (iCaRL)
+ online Elastic Weight Consolidation (oEWC)
+ Synaptic Intelligence
+ Learning without Forgetting
+ Progressive Neural Networks
+ Dark Experience Replay (DER)
+ Dark Experience Replay++ (DER++)

## Datasets

**Class-Il / Task-IL settings**

+ Sequential MNIST
+ Sequential CIFAR-10
+ Sequential Tiny ImageNet

**Domain-IL settings**

+ Permuted MNIST
+ Rotated MNIST

**General Continual Learning setting**

+ MNIST-360

## Results

<table style="
    font-size: 11px;
    white-space: nowrap;
    text-align: right;
    border: 2px solid darkgray;
">
<thead style="/* border-top: 2px solid darkgray; */border-bottom: 2px solid darkgray;">
<tr><th colspan="10" style="text-align:center;">Continual Learning Results</th></tr>
<tr>
    <th rowspan="2" style="text-align:center;">Buffer</th>
    <th rowspan="2" style="text-align:center;">Method</th>
    <th colspan="2" style="text-align:center;">S-CIFAR-10</th>
    <th colspan="2" style="text-align:center;">S-Tiny-ImageNet</th>
    <th style="text-align:center;">P-MNIST</th>
    <th style="text-align:center;">R-MNIST</th>
    <th colspan="2" style="text-align:center;">S-MNIST</th>
</tr>
<tr>
    <th style="text-align:center;">Class-IL</th>
    <th style="text-align:center;">Task-IL</th>
    <th style="text-align:center;">Class-IL</th>
    <th style="text-align:center;">Task-IL</th>
    <th style="text-align:center;">Domain-IL</th>
    <th style="text-align:center;">Domain-IL</th>
    <th style="text-align:center;">Class-IL</th>
    <th style="text-align:center;">Task-IL</th>
</tr></thead>
<tbody style="border-bottom: 2px solid darkgray;">
<tr><td rowspan="6" style="text-align:center;">-</td><td style="text-align:left;">JOINT</td><td>92.20</td><td>98.31</td><td>59.99</td><td>82.04</td><td>94.33</td><td>95.76</td><td>95.57</td><td>99.51</td></tr>
<tr><td style="text-align:left;">SGD</td><td>19.62</td><td>61.02</td><td>7.92</td><td>18.31</td><td>40.70</td><td>67.66</td><td>19.60</td><td>94.94</td></tr>
<tr><td style="text-align:left;">oEWC</td><td>19.49</td><td>68.29</td><td>7.58</td><td>19.20</td><td><b>75.79</b></td><td><b>77.35</b></td><td><b>20.46</b></td><td>98.39</td>
</tr><tr><td style="text-align:left;">SI</td><td>19.48</td><td>68.05</td><td>6.58</td><td>36.32</td><td>65.86</td><td>71.91</td><td>19.27</td><td>96.00</td>
</tr><tr><td style="text-align:left;">LwF</td><td><b>19.61</b></td><td>63.29</td><td><b>8.46</b></td><td>15.85</td><td>-</td><td>-</td><td>19.62</td><td>94.11</td>
</tr><tr><td style="text-align:left;">PNN</td><td>-</td><td><b>95.13</b></td><td>-</td><td><b>67.84</b></td><td>-</td><td>-</td><td>-</td><td><b>99.23</b></td>
</tr></tbody><tbody style="border-bottom: 2px solid darkgray;">
<tr><td rowspan="10" style="text-align:center;">200</td><td style="text-align:left;">ER</td><td>44.79</td><td>91.19</td><td>8.49</td><td>38.17</td><td>72.37</td><td>85.01</td><td>80.43</td><td>97.86</td>
</tr><tr><td style="text-align:left;">MER</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>81.47</td><td>98.05</td>
</tr><tr><td style="text-align:left;">GEM</td><td>25.54</td><td>90.44</td><td>-</td><td>-</td><td>66.93</td><td>80.80</td><td>80.11</td><td>97.78</td>
</tr><tr><td style="text-align:left;">A-GEM</td><td>20.04</td><td>83.88</td><td>8.07</td><td>22.77</td><td>66.42</td><td>81.91</td><td>45.72</td><td>98.61</td>
</tr><tr><td style="text-align:left;">iCaRL</td><td>49.02</td><td>88.99</td><td>7.53</td><td>28.19</td><td>-</td><td>-</td><td>70.51</td><td>98.28</td>
</tr><tr><td style="text-align:left;">FDR</td><td>30.91</td><td>91.01</td><td>8.70</td><td>40.36</td><td>74.77</td><td>85.22</td><td>79.43</td><td>97.66</td>
</tr><tr><td style="text-align:left;">GSS</td><td>39.07</td><td>88.80</td><td>-</td><td>-</td><td>63.72</td><td>79.50</td><td>38.90</td><td>95.02</td>
</tr><tr><td style="text-align:left;">HAL</td><td>32.36</td><td>82.51</td><td>-</td><td>-</td><td>74.15</td><td>84.02</td><td>84.70</td><td>97.96</td>
</tr><tr><td style="text-align:left;"><b>DER</b></td><td>61.93</td><td>91.40</td><td><b>11.87</b></td><td>40.22</td><td>81.74</td><td>90.04</td><td>84.55</td><td><b>98.80</b></td>
</tr><tr><td style="text-align:left;"><b>DER++</b></td><td><b>64.88</b></td><td><b>91.92</b></td><td>10.96</td><td><b>40.87</b></td><td><b>83.58</b></td><td><b>90.43</b></td><td><b>85.61</b></td><td>98.76</td>
</tr></tbody><tbody style="border-bottom: 2px solid darkgray;">
<tr><td rowspan="10" style="text-align:center;">500</td><td style="text-align:left;">ER</td><td>57.74</td><td>93.61</td><td>9.99</td><td>48.64</td><td>80.60</td><td>88.91</td><td>86.12</td><td><b>99.04</b></td>
</tr><tr><td style="text-align:left;">MER</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>88.35</td><td>98.43</td>
</tr><tr><td style="text-align:left;">GEM</td><td>26.20</td><td>92.16</td><td>-</td><td>-</td><td>76.88</td><td>81.15</td><td>85.99</td><td>98.71</td>
</tr><tr><td style="text-align:left;">A-GEM</td><td>22.67</td><td>89.48</td><td>8.06</td><td>25.33</td><td>67.56</td><td>80.31</td><td>46.66</td><td>98.93</td>
</tr><tr><td style="text-align:left;">iCaRL</td><td>47.55</td><td>88.22</td><td>9.38</td><td>31.55</td><td>-</td><td>-</td><td>70.10</td><td>98.32</td>
</tr><tr><td style="text-align:left;">FDR</td><td>28.71</td><td>93.29</td><td>10.54</td><td>49.88</td><td>83.18</td><td>89.67</td><td>85.87</td><td>97.54</td>
</tr><tr><td style="text-align:left;">GSS</td><td>49.73</td><td>91.02</td><td>-</td><td>-</td><td>76.00</td><td>81.58</td><td>49.76</td><td>97.71</td>
</tr><tr><td style="text-align:left;">HAL</td><td>41.79</td><td>84.54</td><td>-</td><td>-</td><td>80.13</td><td>85.00</td><td>87.21</td><td>98.03</td>
</tr><tr><td style="text-align:left;"><b>DER</b></td><td>70.51</td><td>93.40</td><td>17.75</td><td>51.78</td><td>87.29</td><td>92.24</td><td>90.54</td><td>98.84</td>
</tr><tr><td style="text-align:left;"><b>DER++</b></td><td><b>72.70</b></td><td><b>93.88</b></td><td><b>19.38</b></td><td><b>51.91</b></td><td><b>88.21</b></td><td><b>92.77</b></td><td><b>91.00</b></td><td>98.94</td>
</tr></tbody><tbody style="/* border-bottom: 2px solid darkgray; */">
<tr><td rowspan="10" style="text-align:center;">5120</td><td style="text-align:left;">ER</td><td>82.47</td><td><b>96.98</b></td><td>27.40</td><td>67.29</td><td>89.90</td><td>93.45</td><td>93.40</td><td>99.33</td>
</tr><tr><td style="text-align:left;">MER</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>94.57</td><td>99.27</td>
</tr><tr><td style="text-align:left;">GEM</td><td>25.26</td><td>95.55</td><td>-</td><td>-</td><td>87.42</td><td>88.57</td><td>95.11</td><td>99.44</td>
</tr><tr><td style="text-align:left;">A-GEM</td><td>21.99</td><td>90.10</td><td>7.96</td><td>26.22</td><td>73.32</td><td>80.18</td><td>54.24</td><td>98.93</td>
</tr><tr><td style="text-align:left;">iCaRL</td><td>55.07</td><td>92.23</td><td>14.08</td><td>40.83</td><td>-</td><td>-</td><td>70.60</td><td>98.32</td>
</tr><tr><td style="text-align:left;">FDR</td><td>19.70</td><td>94.32</td><td>28.97</td><td>68.01</td><td>90.87</td><td>94.19</td><td>87.47</td><td>97.79</td>
</tr><tr><td style="text-align:left;">GSS</td><td>67.27</td><td>94.19</td><td>-</td><td>-</td><td>82.22</td><td>85.24</td><td>89.39</td><td>98.33</td>
</tr><tr><td style="text-align:left;">HAL</td><td>59.12</td><td>88.51</td><td>-</td><td>-</td><td>89.20</td><td>91.17</td><td>89.52</td><td>98.35</td>
</tr><tr><td style="text-align:left;"><b>DER</b></td><td>83.81</td><td>95.43</td><td>36.73</td><td>69.50</td><td>91.66</td><td>94.14</td><td>94.90</td><td>99.29</td>
</tr><tr><td style="text-align:left;"><b>DER++</b></td><td><b>85.24</b></td><td>96.12</td><td><b>39.02</b></td><td><b>69.84</b></td><td><b>92.26</b></td><td><b>94.65</b></td><td><b>95.30</b></td><td><b>99.47</b></td>
</tr></tbody>
</table>

<table style="
    font-size: 11px;
    white-space: nowrap;
    text-align: right;
    border: 2px solid darkgray;
">
<thead style="
    border-bottom: 2px solid darkgray;
">
<tr><th colspan="9" style="text-align:center;">MNIST-360 - General Continual Learning </th></tr>
<tr><th style="text-align:center;">    JOINT </th><th style="text-align:center;"> SGD </th><th style="text-align:center;"> <b>Buffer</b> </th><th style="text-align:center;"> ER </th><th style="text-align:center;"> MER </th><th style="text-align:center;"> A-GEM-R </th><th style="text-align:center;"> GSS </th><th style="text-align:center;"> <b>DER</b> </th><th style="text-align:center;"> <b>DER++</b></th></tr></thead><tbody>
<tr><td>     </td><td> </td><td> 200 </td><td>  49.27  </td><td> 48.58  </td><td> 28.34  </td><td> 43.92</td><td><b>55.22</b></td><td>54.16</td>                  
</tr><tr><td>     82.98</td><td> 19.09 </td><td> 500  </td><td> 65.04  </td><td> 62.21  </td><td> 28.13  </td><td> 54.45</td><td> 69.11                          </td><td> <b>69.62</b></td>  
</tr><tr><td>    <!--
td--></td><td>  </td><td> 1000  </td><td> 75.18  </td><td> 70.91  </td><td> 29.21  </td><td> 63.84</td><td> 75.97                          </td><td> <b>76.03</b></td>
</tr></tbody></table>

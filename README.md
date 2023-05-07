Download Link: https://assignmentchef.com/product/solved-neural-language-models-assignment-6-solution
<br>
This assignment is a continuation of last week’s assignment. We’ll turn from traditional n-gram based language models to a more advanced form of language modeling using a Recurrent Neural Network. Specifically, we’ll be setting up a character-level recurrent neural network (char-rnn) for short.

Andrej Karpathy, a researcher at OpenAI, has written an excellent blog post about using RNNs for language models, which you should read before beginning this assignment. The title of his blog post is <a href="https://karpathy.github.io/2015/05/21/rnn-effectiveness/">The Unreasonable Effectiveness of Recurrent Neural Networks</a>.

Karpathy shows how char-rnns can be used to generate texts for several fun domains:

<ul>

 <li>Shakespeare plays</li>

 <li>Essays about economics</li>

 <li>LaTeX documents</li>

 <li>Linux source code</li>

 <li>Baby names</li>

</ul>

In this assignment you will follow a Pytorch tutorial code to implement your own char-rnn, and then test it on a dataset of your choice. You will also train on our provided training set, and submit to the leaderboard, where we will measure your model’s complexity on our test set.

<h2 id="part-1-set-up-pytorch">Part 1: Set up Pytorch</h2>

Pytorch is one of the most popular deep learning frameworks in both industry and academia, and learning its use will be invaluable should you choose a career in deep learning. You will be using Pytorch for this assignment, and instead of providing you source code, we ask you to build off a couple Pytorch tutorials.

<h3 id="setup">Setup</h3>

<h4 id="using-miniconda">Using miniconda</h4>

Miniconda is a package, dependency and environment management for python (amongst other languages). It lets you install different versions of python, different versions of various packages in different environments which makes working on multiple projects (with different dependencies) easy.

There are two ways to use miniconda,

<ol>

 <li><strong>Use an existing installation from another user (highly recommended)</strong>: On <code class="highlighter-rouge">biglab</code>, add the following line at the end of your <code class="highlighter-rouge">~/.bashrc</code> file.Then run the following commandIf you run the command <code class="highlighter-rouge">$ which conda</code>, the output should be <code class="highlighter-rouge">/home1/m/mayhew/miniconda3/bin/conda</code>.</li>

 <li><strong>Installing Miniconda from scratch</strong>: On <code class="highlighter-rouge">biglab</code>, run the following commands. Press Enter/Agree to all prompts during installation.After successful installation, running the command <code class="highlighter-rouge">$ which conda</code> should output <code class="highlighter-rouge">/home1/m/$USERNAME/miniconda3/bin/conda</code>.</li>

</ol>

<h4 id="installing-pytorch-and-jupyter">Installing Pytorch and Jupyter</h4>

For this assignment, you’ll be using <a href="http://pytorch.org/">Pytorch</a> and <a href="https://jupyter.org/">Jupyter</a>.

<ol>

 <li>If you followed our recommendation and used the existing miniconda installation from 1. above, you’re good to go. Stop wasting time and start working on the assignment!</li>

 <li>Intrepid students who installed their own miniconda version from 2. above, need to install their own copy of Pytorch and Jupyter. To install Pytorch, run the commandTo check, run python and <code class="highlighter-rouge">import torch</code>. This should run without giving errors. To install jupyter, run the command (it might take a while)Running the command <code class="highlighter-rouge">jupyter --version</code> should yield the version installed.</li>

</ol>

<h4 id="how-to-use-jupyter-notebook">How to use Jupyter notebook</h4>

For this homework, you have the option of using <a href="https://jupyter.org/">jupyter notebook</a>, which lets you interactively edit your code within the web browser. Jupyter reads files in the <code class="highlighter-rouge">.ipynb</code> format. To launch from biglab, do the following.

<ol>

 <li>On <code class="highlighter-rouge">biglab</code>, navigate to the directory with your code files and type <code class="highlighter-rouge">jupyter notebook --port 8888 --no-browser</code>. If you are having token issues, you may need to also add the argument <code class="highlighter-rouge">--NotebookApp.token=''</code>.</li>

 <li>In your local terminal, set up port forward by typing <code class="highlighter-rouge">ssh -N -f -L localhost:8888:localhost:8888 <a href="/cdn-cgi/l/email-protection" class="__cf_email__" data-cfemail="671e08121509060a0227050e000b0605491402061449121702090949020312">[email protected]</a></code>.</li>

 <li>In your local web browser, navigate to <code class="highlighter-rouge">localhost:8888</code>.</li>

</ol>

<h2 id="part-2--classification-using-character-level-recurrent-neural-networks">Part 2: Classification Using Character-Level Recurrent Neural Networks</h2>

<h4 id="follow-the-tutorial-code">Follow the tutorial code</h4>

Read through the tutorial <a href="http://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html">here</a> that builds a char-rnn that is used to classify baby names by their country of origin. While we strongly recommend you carefully read through the tutorial, you will find it useful to build off the released code <a href="https://github.com/spro/practical-pytorch/tree/master/char-rnn-classification/char-rnn-classification.ipynb">here</a>. Make sure you can reproduce the tutorial’s results on the tutorial’s provided baby-name dataset before moving on.

<h4 id="switch-to-city-names-dataset">Switch to city names dataset</h4>

Modify the tutorial code to instead read from the city names dataset that we used in the previous assignment. The tutorial code problematically used the same text file for both training and evaluation. We learned in class about how this is not a great idea. For the city names dataset we provide you separate train and validation sets, as well as a test file for the leaderboard.

All training should be done on the train set and all evaluation (including confusion matrices and accuracy reports) on the validation set. You will need to change the data processing code to get this working. Specifically, you’ll need to modify the code in the 3rd code block to create two variables <code class="highlighter-rouge">category_lines_train</code> and <code class="highlighter-rouge">category_lines_val</code>. In addition, to handle unicode, you might need to replace calls to <code class="highlighter-rouge">open</code> with calls to <code class="highlighter-rouge">codecs.open(filename, "r",encoding='utf-8', errors='ignore')</code>.

Warning: you’ll want to lower the learning rating to 0.002 or less or you might get NaNs when training.

Attribution: the city names dataset is derived from <a href="https://download.maxmind.com/download/geoip/database/LICENSE_WC.txt">Maxmind</a>’s dataset.

<strong>Experimentation and Analysis</strong>

Complete the following analysis on the city names dataset, and include your finding in the report.

<ol>

 <li>Write code to output accuracy on the validation set. Include your best accuracy in the report. (For a benchmark, the TAs were able to get accuracy above 50%) Discuss where your model is making mistakes. Use a confusion matrix plot to support your answer.</li>

 <li>Modify the training loop to periodically compute the loss on the validation set, and create a plot with the training and validation loss as training progresses. Is your model overfitting? Include the plot in your report.</li>

 <li>Experiment with the learning rate. You can try a few different learning rates and observe how this affects the loss. Another common practice is to drop the learning rate when the loss has plateaued. Use plots to explain your experiments and their effects on the loss.</li>

 <li>Experiment with the size of the hidden layer or the model architecture How does this affect validation accuracy?</li>

</ol>

<strong>Leaderboard</strong>

Write code to make predictions on the provided test set. The test set has one unlabeled city name per line. Your code should output a file <code class="highlighter-rouge">labels.txt</code>with one two-letter country code per line. Extra credit will be given to the top leaderboard submissions. Here are some ideas for improving your leaderboard performance:

<ul>

 <li>Play around with the vocabulary (the <code class="highlighter-rouge">all_letters</code> variable), for example modifying it to only include lowercase letters, apostrophe, and the hyphen symbol.</li>

 <li>Test out label smoothing</li>

 <li>Try a more complicated architecture, for example, swapping out the RNN for LSTM or GRU units.</li>

 <li>Use a different initalization for the weights, for example, small random values instead of 0s</li>

</ul>

In your report, describe your final model and training parameters.

<h2 id="part-3-text-generation-using-char-rnn">Part 3: Text generation using char-rnn</h2>

In this section, you will be following more Pytorch tutorial code in order to reproduce Karpathy’s text generation results. Read through the tutorial <a href="http://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html">here</a>, and then download <a href="https://github.com/spro/practical-pytorch/tree/master/char-rnn-generation">this ipython notebook</a> to base your own code on.

You will notice that the code is quite similar to that of the classification problem. The biggest difference is in the loss function. For classification, we run the entire sequence through the RNN and then impose a loss only on the final class prediction. For the text generation task, we impose a loss at each step of the RNN on the predicted character. The classes in this second task are the possible characters to predict.

<h4 id="experimenting-with-your-own-dataset">Experimenting with your own dataset</h4>

Be creative! Pick some dataset that interests you. Here are some ideas:

<ul>

 <li><a href="https://raw.githubusercontent.com/rdeese/tunearch-corpus/master/all-abcs.txt">ABC music format</a></li>

 <li><a href="https://github.com/ryanmcdermott/trump-speeches">Donald Trump speeches</a></li>

 <li><a href="https://www.gutenberg.org/cache/epub/29765/pg29765.txt">Webster dictionary</a></li>

 <li><a href="https://www.gutenberg.org/files/31100/31100.txt">Jane Austen novels</a></li>

</ul>

<h4 id="in-your-report">In your report:</h4>

Include a sample of the text generated by your model, and give a qualitative discussion of the results. Where does it do well? Where does it seem to fail? Report perplexity on a couple validation texts that are similar and different to the training data. Compare your model’s results to that of an n-gram language model.

<h2 id="deliverables">Deliverables</h2>

<h2 id="recommended-readings">Recommended readings</h2>

<table>

 <tbody>

  <tr>

   <td><a href="https://web.stanford.edu/~jurafsky/slp3/8.pdf">Neural Nets and Neural Language Models.</a> Dan Jurafsky and James H. Martin. Speech and Language Processing (3rd edition draft) .</td>

  </tr>

  <tr>

   <td><a href="https://karpathy.github.io/2015/05/21/rnn-effectiveness/">The Unreasonable Effectiveness of Recurrent Neural Networks.</a> Andrej Karpathy. Blog post. 2015.</td>

  </tr>

  <tr>

   <td><a href="http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf">A Neural Probabilistic Language Model (longer JMLR version).</a> Yoshua Bengio, Réjean Ducharme, Pascal Vincent and Christian Jauvin. Journal of Machine Learning Research 2003</td>

  </tr>

 </tbody>

</table>

5/5 - (2 votes)

Here are the materials that you should download for this assignment:

<ul>

 <li><a href="http://computational-linguistics-class.org/downloads/hw5/cities_train.zip">training data for text classification task</a>.</li>

 <li><a href="http://computational-linguistics-class.org/downloads/hw5/cities_val.zip">dev data for text classification task</a>.</li>

 <li><a href="http://computational-linguistics-class.org/downloads/hw5/cities_test.txt">test file for leaderboard</a></li>

</ul>

<pre class="highlight"><code>export PATH="/home1/m/mayhew/miniconda3/bin:$PATH"</code></pre>

<pre class="highlight"><code>source ~/.bashrc</code></pre>

<pre class="highlight"><code>$ wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh$ chmod +x Miniconda3-latest-Linux-x86_64.sh$ bash Miniconda3-latest-Linux-x86_64.sh</code></pre>

<pre class="highlight"><code>conda install pytorch-cpu torchvision -c pytorch</code></pre>

<pre class="highlight"><code>conda install jupyter</code></pre>

Download the city names dataset.

<ul>

 <li><a href="http://computational-linguistics-class.org/downloads/hw5/cities_train.zip">training sets</a></li>

 <li><a href="http://computational-linguistics-class.org/downloads/hw5/cities_val.zip">validation set</a></li>

 <li><a href="http://computational-linguistics-class.org/downloads/hw5/cities_test.txt">test file for leaderboard</a></li>

</ul>

Here are the deliverables that you will need to submit:

<ul>

 <li>writeup.pdf</li>

 <li>code (.zip). It should be written in Python 3 and include a README.txt briefly explaining how to run it.</li>

 <li><code class="highlighter-rouge">labels.txt</code> predictions for leaderboard.</li>
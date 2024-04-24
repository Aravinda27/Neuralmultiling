# NeuralMultiling: A Novel Neural Architecture Search for Smartphone based Multilingual Speaker Verification
# Summary
Multilingual speaker verification introduces the challenge of verifying a speaker in multiple languages. Existing systems were built using i-vector/x-vector approaches along with Bi-LSTMs, which were trained to discriminate speakers, irrespective of the language. Instead of exploring the design space manually, we propose a neural architecture search for multilingual speaker verification suitable for mobile devices, called NeuralMultiling. First, our algorithm searches for an optimal operational combination of neural cells with different architectures for normal cells and reduction cells and then derives a CNN model by stacking neural cells. Using the derived architecture, we performed two different studies:1) language agnostic condition and 2) interoperability between languages and devices on the publicly available Multilingual Audio-Visual Smartphone (MAVS) dataset. The experimental results suggest that the derived architecture significantly outperforms the existing Autospeech method by a 5-6% reduction in the Equal Error Rate (EER) with fewer model parameters.

# Spawning different architectures for Normal and Reduction cells using Proposed algorithm

<div align="center">
  <img src="figures/algorithm.drawio.png" alt="Image Description">
  <p><em> Left: Algorithm of Autospeech which spawns same architecture for Normal and Reduction cell Right: Algorithm for the proposed NeuralMultiling which spwans different architecture for Normal and Reduction cell</em></p>
</div>

# Requirements
  Python=3.9 <br>
Pytorch>=1.0 <br>
Other Requirements=pip install -r requirements

## Dataset
You will need Multilingual Audio-Visual Smartphone (MAVS) database. 

## Running the code
- dataprocess: <br>
    python data_preprocess.py /path/to/MAVS


# Results 
# Language Agnostic Condition
<div align="center">
  <img src="figures/LA.png" alt="Image Description">
  <p><em> </em></p>
</div>

# Interoperability between Languages and Devices
<div align="center">
  <img src="figures/proposed.png" alt="Image Description">
  <p><em> </em></p>
</div>
<div align="center">
  <img src="figures/Autospeech.png" alt="Image Description">
  <p><em> </em></p>
</div>

# Visualisation of derieved architectures for Normal and Reduction cell.
<div align="center">
  <table>
    <tr>
      <td style="text-align:center">
        <img src="figures/Normal_cell_new_page-0001.jpg" alt="Image 1" width="500" />
        <br />
        <em>Normal cell</em>
      </td>
      <td style="text-align:center">
        <img src="figures/Reduction_cell_new_page-0001.jpg" alt="Image 2" width="500" />
        <br />
        <em>Reduction cell</em>
      </td>
    </tr>
  </table>
</div>




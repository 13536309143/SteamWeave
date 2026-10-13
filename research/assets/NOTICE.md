# Research asset provenance

`minimax_semantics.json` contains structured, evidence-validated model outputs for the supplied simulated inspection reports. It contains no credentials. Models and prompt version are recorded per entry. The MiniMax M2.7 standard and high-speed aliases share the legacy namespace used by the original deployment; other provider/model/endpoint combinations use isolated namespaces.

`glove_vocab.json` contains 129 corpus words and their 100-dimensional vectors from the official `glove.6B.100d.txt` distribution. 127 words are covered; two uncovered contractions retain a missing-vector flag and are initialized from the configured seed when training the reconstruction. These are a subset of fixed pretrained vectors, not vectors trained on target labels.

GloVe upstream: [Stanford GloVe](https://nlp.stanford.edu/projects/glove/). Its official page distributes pretrained vectors under [PDDL 1.0](https://opendatacommons.org/licenses/pddl/1-0/).

Pennington, J., Socher, R., and Manning, C. (2014). *GloVe: Global Vectors for Word Representation.* EMNLP.

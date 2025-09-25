
"""
Author: Yuetong Fang
Modified from: https://arxiv.org/abs/2111.01456
"""

from rockpool.nn.modules import (
    TorchModule,
    LinearTorch,
    LIFTorch,
    ExpSynTorch,
    LIFMembraneExodus,
)

from rockpool.nn.modules.sinabs.lif_exodus import LIFSlayer
from rockpool.parameters import Parameter, State, SimulationParameter, Constant
from rockpool.nn.modules.torch.lif_torch import (
    LIFBaseTorch,
    StepPWL,
    PeriodicExponential,
)
from rockpool.graph import AliasConnection, GraphHolder, connect_modules

import torch

from typing import List, Tuple, Union, Callable, Optional, Type
from rockpool.typehints import P_tensor
from quantization import *

__all__ = ["WaveSenseBlock", "WaveSenseNet_Hardware"]


class WaveSenseBlock(TorchModule):
    """
    Implements a single WaveSenseBlock
                          ▲
           To next block  │       ┌─────────────────┐
       ┌──────────────────┼───────┤ WaveSenseBlock  ├───┐
       │                  │       └─────────────────┘   │
       │ Residual path   .─.                            │
       │    ─ ─ ─ ─ ─ ─▶( + )                           │
       │    │            `─'                            │
       │                  ▲                             │
       │    │             │                             │
       │               .─────.                          │
       │    │         ( Spike )                         │
       │               `─────'                          │
       │    │             ▲                             │
       │                  │                             │
       │    │       ┌──────────┐                        │
       │            │  Linear  │                        │
       │    │       └──────────┘         Skip path      │    Skip
       │                  ▲       ┌──────┐    .─────.   │ connections
       │    │             ├──────▶│Linear│──▶( Spike )──┼──────────▶
       │                  │       └──────┘    `─────'   │
       │    │          .─────.                          │
       │              ( Spike )                         │
       │    │          `─────'                          │
       │                 ╲┃╱                            │
       │    │             ┃ Dilation                    │
       │            ┌──────────┐                        │
       │    │       │  Linear  │                        │
       │            └──────────┘                        │
       │    │             ▲                             │
       │     ─ ─ ─ ─ ─ ─ ─│                             │
       └──────────────────┼─────────────────────────────┘
                          │ From previous block
                          │
    """

    def __init__(
        self,
        Nchannels: int = 16,
        Nskip: int = 32,
        dilation: int = None,
        kernel_size: int = 2,
        bias: P_tensor = Constant(0.0),
        tau_mem: float = Constant(10e-3),
        base_tau_syn: float = Constant(10e-3),
        threshold: float = Constant(1.0),
        neuron_model: Type = LIFTorch,
        dt: float = 1e-3,
        *args,
        **kwargs,
    ):
        """
        Implementation of the WaveBlock as used in the WaveSense model. It received (Nchannels) input channels and outputs (Nchannels, Nskip) channels.

        Args:
            :param int Nchannels:           Dimensionality of the residual connection. Default: ``16``
            :param int Nskip:               Dimensionality of the skip connection. Default: ``32``
            :param int dilation:            Determines the synaptic time constant of the dilation layer $dilation * base_tau_syn$. Default: ``None``
            :param int kernel_size:         Number of synapses the time dilation layer in the WaveBlock. Default: ``2``
            :param P_tensor bias:           Bias for the network to train. Default: No trainable bias
            :param float tau_mem:           Membrane potential time constant of all neurons in WaveSense. Default: 10ms
            :param float base_tau_syn:      Base synaptic time constant. Each synapse has this time constant, except the second synapse in the dilation layer which caclulates the time constant as $dilations * base_tau_syn$. Default: 10ms
            :param float threshold:         Threshold of all spiking neurons. Default: `0.`
            :param Type neuron_model: Neuron model to use. Either :py:class:`.LIFTorch` as standard LIF implementation, :py:class:`.LIFBitshiftTorch` for hardware compatibility or :py:class:`.LIFExodus` for speedup
            :param float dt:                Temporal resolution of the simulation. Default: 1ms
        """
        # - Initialise superclass
        super().__init__(
            shape=(Nchannels, Nchannels),
            spiking_input=True,
            spiking_output=True,
            *args,
            **kwargs,
        )

        # - Add parameters
        self.neuron_model: Union[Callable, SimulationParameter] = SimulationParameter(
            neuron_model
        )
        """ Neuron model used by this WaveSense network """

        # - Dilation layers
        tau_syn = torch.arange(0, dilation * kernel_size, dilation) * base_tau_syn
        tau_syn = torch.clamp(tau_syn, base_tau_syn, tau_syn.max()).repeat(Nchannels, 1)

        self.lin1 = LinearTorch(
            shape=(Nchannels, Nchannels * kernel_size), has_bias=False
        )
        with torch.no_grad():
            self.lin1.weight.data = self.lin1.weight.data * dt / tau_syn.max()

        self.spk1 = self.neuron_model(
            shape=(Nchannels * kernel_size, Nchannels),
            tau_mem=Constant(tau_syn.max().item()),
            tau_syn=Constant(tau_syn),
            bias=bias,
            threshold=Constant(threshold),
            has_rec=False,
            w_rec=None,
            noise_std=0,
            spike_generation_fn=PeriodicExponential,
            #max_spikes_per_dt = torch.tensor(1.0),
            learning_window=0.5,
            dt=dt,
        )

        # - Remapping output layers
        self.lin2_res = LinearTorch(shape=(Nchannels, Nchannels), has_bias=False)
        with torch.no_grad():
            self.lin2_res.weight.data = self.lin2_res.weight.data * dt / tau_syn.min()

        self.spk2_res = self.neuron_model(
            shape=(Nchannels, Nchannels),
            tau_mem=Constant(tau_mem),
            tau_syn=Constant(tau_syn.min()),
            bias=bias,
            threshold=Constant(threshold),
            has_rec=False,
            w_rec=None,
            noise_std=0,
            spike_generation_fn=PeriodicExponential,
            #max_spikes_per_dt = torch.tensor(1.0),
            learning_window=0.5,
            dt=dt,
        )

        # - Skip output layers
        self.lin2_skip = LinearTorch(shape=(Nchannels, Nskip), has_bias=False)
        with torch.no_grad():
            self.lin2_skip.weight.data = self.lin2_skip.weight.data * dt / tau_syn.min()

        self.spk2_skip = self.neuron_model(
            shape=(Nskip, Nskip),
            tau_mem=Constant(tau_mem),
            tau_syn=Constant(tau_syn.min()),
            bias=bias,
            threshold=Constant(threshold),
            #max_spikes_per_dt = torch.tensor(1.0),
            dt=dt,
        )

        # - Internal record dictionary
        self._record = True
        self._record_dict = {}

    def forward(self, data: torch.tensor) -> Tuple[torch.tensor, dict, dict]:
        # Expecting data to be of the format (batch, time, Nchannels)
        (n_batches, t_sim, Nchannels) = data.shape

        # - Pass through dilated weight layer
        out, _, self._record_dict["lin1"] = self.lin1(data, record=self._record)
        self._record_dict["lin1_output"] = out if self._record else []

        # - Pass through dilated spiking layer
        hidden, _, self._record_dict["spk1"] = self.spk1(
            out, record=self._record
        )  # (t_sim, n_batches, Nchannels)
        self._record_dict["spk1_output"] = out if self._record else []

        # - Pass through output linear weights
        out_res, _, self._record_dict["lin2_res"] = self.lin2_res(
            hidden, record=self._record
        )
        self._record_dict["lin2_res_output"] = out_res if self._record else []

        # - Pass through output spiking layer
        out_res, _, self._record_dict["spk2_res"] = self.spk2_res(
            out_res, record=self._record
        )
        self._record_dict["spk2_res_output"] = out_res if self._record else []

        # - Hidden -> skip outputs
        out_skip, _, self._record_dict["lin2_skip"] = self.lin2_skip(
            hidden, record=self._record
        )
        self._record_dict["lin2_skip_output"] = out_skip if self._record else []

        # - Pass through skip output spiking layer
        out_skip, _, self._record_dict["spk2_skip"] = self.spk2_skip(
            out_skip, record=self._record
        )
        self._record_dict["spk2_skip_output"] = out_skip if self._record else []

        # - Combine output and residual connections (pass-through)
        res_out = out_res + data

        return res_out, out_skip

    def evolve(self, input, record: bool = False):
        self._record = record

        # - Use super-class evolve
        output, new_state, _ = super().evolve(input, self._record)

        # - Get state record from property
        record_dict = self._record_dict if self._record else {}

        return output, new_state, record_dict

    def as_graph(self):
        mod_graphs = []

        for mod in self.modules().values():
            mod_graphs.append(mod.as_graph())

        connect_modules(mod_graphs[0], mod_graphs[1])
        connect_modules(mod_graphs[1], mod_graphs[2])
        connect_modules(mod_graphs[2], mod_graphs[3])  # skip_res
        connect_modules(mod_graphs[1], mod_graphs[4])
        connect_modules(mod_graphs[4], mod_graphs[5])  # skip_add

        AliasConnection(
            mod_graphs[0].input_nodes,
            mod_graphs[3].output_nodes,
            name=f"residual_loop",
            computational_module=None,
        )

        multiple_out = mod_graphs[3].output_nodes
        multiple_out.extend(mod_graphs[5].output_nodes)

        return GraphHolder(
            mod_graphs[0].input_nodes,
            multiple_out,
            f"{type(self).__name__}_{self.name}_{id(self)}",
            self,
        )


class WaveSenseNet_Hardware(TorchModule):
    """
    Implement a WaveSense network
                                                         Threshold
                                                         on output
                                                .───────.
                                               (Low-pass )────▶
                                                `───────'
                                                    ▲
                                                    │
                                              ┌──────────┐
                                              │  Linear  │
                                              └──────────┘
                                                    ▲
                                                    │
                                                 .─────.
                                                ( Spike )
    ┌──────────────────────┐         Skip        `─────'
    │                      ├┐      outputs          ▲
    │ WaveSenseBlock stack │├┬───┐                  │
    │                      ││├┬──┤      .─.   ┌──────────┐
    └┬─────────────────────┘││├──┴┬───▶( + )─▶│  Linear  │
     └┬─────────────────────┘││───┘     `─'   └──────────┘
      └┬─────────────────────┘│
       └──────────────────────┘
                   ▲
                   │
                .─────.
               ( Spike )
                `─────'
                   ▲
                   │
             ┌──────────┐
             │  Linear  │
             └──────────┘
                   ▲  Spiking
                   │   input
    """

    def __init__(
        self,
        dilations: List,
        n_classes: int = 2,
        n_channels_in: int = 16,
        n_channels_res: int = 16,
        n_channels_skip: int = 32,
        n_hidden: int = 32,
        kernel_size: int = 2,
        bias: P_tensor = Constant(0.0),
        smooth_output: bool = True,
        tau_mem: float = Constant(20e-3),
        base_tau_syn: float = Constant(20e-3),
        tau_lp: float = Constant(20e-3),
        threshold: float = Constant(1.0),
        threshold_out: float = Constant(1.0),
        neuron_model: Type = LIFTorch,
        neuron_model_out: Type = None,
        dt: float = 1e-3,
        *args,
        **kwargs,
    ):
        """
        Implementation of the WaveSense network as described in https://arxiv.org/abs/2111.01456.

        Args:
            :param List dilations:          List of dilations which determines the number of WaveBlockes used and the synaptic time constant of the dilation layer $dilations * base_tau_syn$.
            :param int n_classes:           Output dimensionality, usually one per class. Default: ``2``
            :param int n_channels_in:       Input dimensionality / number of input features. Default: ``16``
            :param int n_channels_res:      Dimensionality of the residual connection in each WaveBlock. Default: ``16``
            :param int n_channels_skip:     Dimensionality of the skip connection. Default: ``32``
            :param int n_hidden:            Number of neurons in the hidden layer of the readout. Default: ``32``
            :param int kernel_size:         Number of synapses the dilated layer in the WaveBlock. Default: ``2``
            :param P_tensor bias:           Bias for the network to train. Default: No trainable bias
            :param bool smooth_output:      If the output of the network is smoothed with an exponential kernel. Default: ``True``, use a low-pass filter on the output.
            :param float tau_mem:           Membrane potential time constant of all neurons in WaveSense. Default: 20ms
            :param float base_tau_syn:      Base synaptic time constant. Each synapse has this time constant, except the second synapse in the dilation layer which caclulates the time constant as $dilations * base_tau_syn$. Default: 20ms
            :param float tau_lp:            Time constant of the smooth output. Default: 20ms
            :param float threshold:         Threshold of all neurons in WaveSense. Default: `1.0`
            :param Type neuron_model: Neuron model to use. Either :py:class:`.LIFTorch` as standard LIF implementation, :py:class:`.LIFBitshiftTorch` for hardware compatibility or :py:class:`.LIFExodus` for speedup. Default: :py:class:`.LIFTorch`
            :param float dt:                Temporal resolution of the simulation. Default: 1ms
        """

        # - Determine network shape and initialise
        shape = (n_channels_in, n_classes)

        super().__init__(
            shape=shape, spiking_input=True, spiking_output=True, *args, **kwargs
        )

        self.n_classes = n_classes
        self.n_channels_res = n_channels_res
        self.n_channels_skip = n_channels_skip
        self.n_hidden = n_hidden

        self.neuron_model = neuron_model
        self.neuron_model_out = (
            neuron_model_out if neuron_model_out is not None else neuron_model
        )

        if not issubclass(self.neuron_model, LIFBaseTorch) or not issubclass(
            self.neuron_model_out, LIFBaseTorch
        ):
            raise ValueError(
                "Only `LIFBaseTorch` subclasses are permitted for Wavesense neuron models."
            )

        # - Input mapping layers
        self.lin1 = LinearTorch(shape=(n_channels_in, n_channels_res), has_bias=False)
        with torch.no_grad():
            self.lin1.weight.data = self.lin1.weight.data * dt / base_tau_syn

        self.spk1 = self.neuron_model(
            shape=(n_channels_res, n_channels_res),
            tau_mem=Constant(tau_mem),
            tau_syn=Constant(base_tau_syn),
            bias=bias,
            threshold=Constant(threshold),
            has_rec=False,
            w_rec=None,
            noise_std=0,
            spike_generation_fn=PeriodicExponential,
            # max_spikes_per_dt = torch.tensor(1.0),
            learning_window=0.5,
            dt=dt,
        )

        # - WaveBlock layers
        self._num_dilations = len(dilations)
        self.wave_blocks = []
        for i, dilation in enumerate(dilations):
            wave = WaveSenseBlock(
                n_channels_res,
                n_channels_skip,
                dilation=dilation,
                kernel_size=kernel_size,
                bias=bias,
                tau_mem=Constant(tau_mem),
                base_tau_syn=Constant(base_tau_syn),
                threshold=Constant(threshold),
                neuron_model=neuron_model,
                dt=dt,
            )
            self.__setattr__(f"wave{i}", wave)
            self.wave_blocks.append(wave)

        # Dense readout layers
        self.hidden = QuantLinear(shape=(n_channels_skip, n_hidden), has_bias=False)
        with torch.no_grad():
            self.hidden.weight.data = self.hidden.weight.data * dt / base_tau_syn

        #self.filter = nn.ReLU()

        self.spk2 = self.neuron_model(
            #shape=(1, 1),
            shape=(n_hidden, n_hidden),
            leak_mode = "taus",
            tau_mem=6.83924951, #hardware param
            tau_syn=0.005, #hardware param
            bias=bias,
            threshold=Constant(threshold),
            has_rec=False,
            w_rec=None,
            noise_std=0,
            spike_generation_fn=PeriodicExponential,
            learning_window=0.5,
            max_spikes_per_dt = torch.tensor(1.0),
            dt=dt,
        )

        self.readout = QuantLinear(shape=(n_hidden, n_classes), has_bias=False)
        #self.readout = LinearTorch(shape=(n_hidden, n_classes), has_bias=False)
        with torch.no_grad():
            self.readout.weight.data = self.readout.weight.data * dt / tau_lp

        if self.neuron_model_out is not LIFMembraneExodus:
            self.spk_out = self.neuron_model_out(
                shape=(n_classes, n_classes),
                leak_mode = "taus",
                tau_mem=6.83924951,
                tau_syn=0.005,
                bias=bias,
                threshold=Constant(threshold_out),
                has_rec=False,
                w_rec=None,
                noise_std=0,
                spike_generation_fn=PeriodicExponential,
                max_spikes_per_dt = torch.tensor(1.0),
                learning_window=0.5,
                dt=dt,
            )
        else:
            self.spk_out = self.neuron_model_out(
                #shape=(1, 1),
                shape=(n_classes, n_classes),
                leak_mode = "taus_no_train",
                tau_mem=6.83924951, #hardware param
                tau_syn=0.005, #hardware param
                w_rec=None,
                spike_generation_fn=PeriodicExponential,
                max_spikes_per_dt = torch.tensor(1.0),
                learning_window=0.5,
                dt=dt,
            )

        # - Record dt
        self.dt = SimulationParameter(dt)
        """ float: Time-step in seconds """

        # Dictionary for recording state
        self._record = True
        self._record_dict = {}

    def forward(self, data: torch.Tensor):
        # Expected data shape
        (n_batches, t_sim, n_channels_in) = data.shape

        # - Input mapping layers
        out, _, self._record_dict["lin1"] = self.lin1(data, record=self._record)
        self._record_dict["lin1_output"] = out if self._record else []

        # Pass through spiking layer
        out, _, self._record_dict["spk1"] = self.spk1(
            out, record=self._record
        )  # (t_sim, n_batches, Nchannels)
        self._record_dict["spk1_output"] = out if self._record else []

        # Pass through each wave block in turn
        skip = 0
        for wave_index, wave_block in enumerate(self.wave_blocks):
            (out, skip_new), _, self._record_dict[f"wave{wave_index}"] = wave_block(
                out, record=self._record
            )
            self._record_dict[f"wave{wave_index}_output"] = out if self._record else []
            skip = skip_new + skip

        # Dense layers
        out, _, self._record_dict["hidden"] = self.hidden(skip, record=self._record)
        #out = self.filter(out)
        self._record_dict["hidden_output"] = out if self._record else []

        out, _, self._record_dict["spk2"] = self.spk2(out, record=self._record)
        self._record_dict["spk2_output"] = out if self._record else []

        # Final readout layer
        out, _, self._record_dict["readout"] = self.readout(out, record=self._record)
        self._record_dict["readout_output"] = out if self._record else []

        out, _, self._record_dict["spk_out"] = self.spk_out(out, record=self._record)

        return out

    def evolve(self, input_data, record: bool = False):
        # - Store "record" state
        self._record = record

        # - Evolve network
        output, new_state, _ = super().evolve(input_data, record=self._record)

        # - Get recording dictionary
        record_dict = self._record_dict if self._record else {}

        # - Return
        return output, new_state, record_dict

    def as_graph(self):
        # - Convert all modules to graph representation
        mod_graphs = {k: m.as_graph() for k, m in self.modules().items()}

        # - Connect modules
        connect_modules(mod_graphs["lin1"], mod_graphs["spk1"])
        connect_modules(mod_graphs["spk1"], mod_graphs["wave0"])

        for i in range(self._num_dilations - 1):
            connect_modules(
                mod_graphs[f"wave{i}"],
                mod_graphs[f"wave{i+1}"],
                range(self.n_channels_res),
            )

            AliasConnection(
                mod_graphs[f"wave{i}"].output_nodes[self.n_channels_res :],
                mod_graphs[f"wave{i+1}"].output_nodes[self.n_channels_res :],
                name="skip_add",
                computational_module=None,
            )
        if self._num_dilations == 1:
            connect_modules(
                mod_graphs[f"wave{0}"],
                mod_graphs["hidden"],
                range(self.n_channels_res, self.n_channels_res + self.n_channels_skip),
                None,
            )
        else:
            connect_modules(
                mod_graphs[f"wave{i+1}"],
                mod_graphs["hidden"],
                range(self.n_channels_res, self.n_channels_res + self.n_channels_skip),
                None,
            )
        connect_modules(mod_graphs["hidden"], mod_graphs["spk2"])
        connect_modules(mod_graphs["spk2"], mod_graphs["readout"])
        connect_modules(mod_graphs["readout"], mod_graphs["spk_out"])

        return GraphHolder(
            mod_graphs["lin1"].input_nodes,
            mod_graphs["spk_out"].output_nodes,
            f"{type(self).__name__}_{self.name}_{id(self)}",
            self,
        )

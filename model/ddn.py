"""
Module: Deep Differential Network (DDN)
=======================================

Purpose:
    Implements a "Differential Machine Learning" architecture. unlike standard regression
    models that only minimize the error between Predicted Price vs. Actual Price,
    this model minimizes a combined loss function of:
    1. Price Error (Values)
    2. Gradient Error (Sensitivities/Greeks)

    This approach, often called "Sobolev Training," allows the network to learn the 
    financial pricing function significantly faster and with far less data because 
    calculating the derivative w.r.t inputs forces the model to learn the geometric 
    structure of the option surface, not just point estimates.

Key Mechanisms:
    - Twin Gradient Tapes: Used in the training step. The inner tape calculates 
      the Greeks (dPrice/dInput), and the outer tape calculates the updates 
      for the network weights (dLoss/dWeights).
    - Softplus Activation: Ensures the output option price is strictly positive, 
      preserving financial arbitrage constraints.
"""

import tensorflow as tf                         # for the differential neural network
from tensorflow.keras import layers, Model      # for the network architecture
from typing import Tuple, Dict                  # for type hints

class DeepDifferentialNetwork(Model):
    """
    A Feed-Forward Neural Network customized for Differential Learning.
    
    Attributes:
        hidden_layers (list): Stack of Dense layers with Swish activation.
        output_price (layer): Final projection layer ensuring positive pricing.
    """
    def __init__(self, num_hidden: int = 4, neurons: int = 64, dropout: float = 0.2, activation: str = 'swish', **kwargs):
        """
        Initializes a Deep Differential Network.

        Parameters:
            num_hidden (int): Number of hidden layers. Defaults to 4.
            neurons (int): Number of neurons in each hidden layer. Defaults to 64.
            dropout (float): Dropout rate for each layer. Defaults to 0.2.
            activation (str): Activation function for each layer. Defaults to 'swish'.
            **kwargs: Additional keyword arguments for the parent class.
        """
        super().__init__(**kwargs)
        self.hidden_layers = []
        
        # --- 1. Configurable Architecture ---
        # We build a standard MLP stack. 
        for _ in range(num_hidden):
            self.hidden_layers.append(layers.Dense(neurons, activation=activation))
            self.hidden_layers.append(layers.Dropout(dropout))
            
        # --- 2. Financial Constraints ---
        # Output layer uses 'softplus' (smooth approximation of ReLU: log(1+e^x)).
        # This guarantees Price > 0, which is a hard constraint for Option Pricing.
        self.output_price = layers.Dense(1, activation='softplus', name='price_out')

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Forward pass through the Deep Differential Network.

        Parameters:
            x (tf.Tensor): Input tensor to the network.
            training (bool): If True, apply dropout and other training effects.

        Returns:
            tf.Tensor: Output tensor of the network, representing a price.
        """
        for layer in self.hidden_layers:
            x = layer(x, training=training)
        return self.output_price(x)

    def train_step(self, data: Tuple[tf.Tensor, tf.Tensor]) -> Dict[str, float]:
        """
        A single training step for the Deep Differential Network.

        Parameters:
            data (Tuple[tf.Tensor, tf.Tensor]): Input data, consisting of features and labels.

        Returns:
            dict: A dictionary containing the loss, price loss, and gradient loss metrics.
        """
        # Unpack data: x = features, y = [Target Price, Target Greeks...]
        x, y = data
        
        # Slice labels based on data generation logic
        y_price = y[:, 0:1] # Column 0 is the option price
        y_grads = y[:, 1:]  # Columns 1+ are the analytic Greeks (from QuantLib)
        
        # --- 1. The Outer Tape ---
        # Tracks operations for updating Model Weights (Standard Backprop)
        with tf.GradientTape() as tape:
            
            # --- 2. The Inner Tape ---
            # Tracks operations for computing Greeks (dPrice/dInputs)
            with tf.GradientTape() as inner:
                # We must explicitly 'watch' the input tensor 'x' to calculate derivates w.r.t it.
                inner.watch(x)
                pred_price = self(x, training=True)
            
            # Compute gradients of the Output Price w.r.t Input Features (x).
            # This yields the "Neural Greeks".
            full_grads = inner.gradient(pred_price, x)
            
            # Slice: We only care about gradients for the first 5 params 
            # (Kappa, Lambda, Sigma, Rho, v0). 
            # Gradients w.r.t r, tau, moneyness might be ignored or tracked differently.
            pred_grads = full_grads[:, :5] 
            
            # --- 3. Loss Calculation ---
            # Price Loss (MSE)
            l_p = tf.reduce_mean(tf.square(y_price - pred_price))
            
            # Gradient Loss (MSE of Analytic Greeks vs Neural Greeks)
            # This is the "Sobolev" regularization term.
            l_g = tf.reduce_mean(tf.square(y_grads - pred_grads))
            
            # Total Loss = Alpha * Price_Loss + Gradient_Loss
            # We weight price higher (10.0) to ensure the fundamental value is anchored,
            # while the gradient loss shapes the curve locally.
            loss = (10.0 * l_p) + l_g

        # --- 4. Optimization Step ---
        # Compute gradients of the Loss w.r.t Model Weights (using the Outer Tape)
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        # Return metrics for the progress bar
        return {"loss": loss, "l_p": l_p, "l_g": l_g}

    def test_step(self, data: Tuple[tf.Tensor, tf.Tensor]) -> Dict[str, float]:
        """
        A single testing step for the Deep Differential Network.

        Parameters:
            data (Tuple[tf.Tensor, tf.Tensor]): Input data, consisting of features and labels.

        Returns:
            dict: A dictionary containing the loss, price loss, and gradient loss metrics.
        """
        # Unpack data: x = features, y = [Target Price, Target Greeks...]
        x, y = data
        
        # Slice labels based on data generation logic
        y_price = y[:, 0:1] # Column 0 is the option price
        y_grads = y[:, 1:]  # Columns 1+ are the analytic Greeks (from QuantLib)
        
        # We still need one tape to calculate the Neural Greeks for validation error
        with tf.GradientTape() as inner:
            inner.watch(x)
            pred_price = self(x, training=False)
        
        # Compute gradients of the Output Price w.r.t Input Features (x).
        # This yields the "Neural Greeks".
        full_grads = inner.gradient(pred_price, x)
        
        # Slice: We only care about gradients for the first 5 params 
        # (Kappa, Lambda, Sigma, Rho, v0). 
        # Gradients w.r.t r, tau, moneyness might be ignored or tracked differently.
        pred_grads = full_grads[:, :5]
        
        # Price Loss (MSE)
        l_p = tf.reduce_mean(tf.square(y_price - pred_price))
        
        # Gradient Loss (MSE of Analytic Greeks vs Neural Greeks)
        # This is the "Sobolev" regularization term.
        l_g = tf.reduce_mean(tf.square(y_grads - pred_grads))
        
        # Total Loss = Alpha * Price_Loss + Gradient_Loss
        loss = (10.0 * l_p) + l_g
        
        return {"loss": loss, "l_p": l_p, "l_g": l_g}
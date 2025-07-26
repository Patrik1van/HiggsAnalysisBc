import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, LeakyReLU, Reshape, Dropout, Flatten
from tensorflow.keras.optimizers import AdamW


class Generator:
    """
    The Generator model for the GAN.
    It takes a random noise vector (latent dimension) and outputs a
    tensor of the desired shape (36, 1).
    """
    def __init__(self, latent_dim=100, output_shape=(36, 1), n_units=128, **kwargs):
        self.latent_dim = latent_dim
        self.output_shape = output_shape
        self.n_units = n_units
        self.optimizer = AdamW(learning_rate=kwargs.pop('learning_rate', 0.0001), 
                               weight_decay=kwargs.pop('weight_decay', 1e-4))
        self.num_layers = kwargs.pop('n_layers', 7)
        
        self.model = self._build_model()

    def _build_model(self):
        """Builds the generator architecture."""
        noise_input = Input(shape=(self.latent_dim,))
        
        # Start with a dense layer
        x = noise_input
        for _ in range(self.num_layers):
            x = Dense(self.n_units, use_bias=False)(x)
            x = BatchNormalization()(x)
            x = LeakyReLU(negative_slope=0.2)(x)

        # Output layer
        x = Dense(self.output_shape[0], activation='tanh')(x) # Tanh is common for GANs
        output_tensor = Reshape(self.output_shape)(x)
        
        model = Model(noise_input, output_tensor, name="generator")
        print("--- Generator Architecture ---")
        model.summary()
        return model

    def __call__(self, inputs):
        """Makes the class instance callable."""
        return self.model(inputs)
    
class Discriminator:
    """
    The Discriminator model for the GAN.
    It takes a tensor shaped like the real data (36, 1) and outputs a
    single value indicating whether the input is real (close to 1) or fake (close to 0).
    """
    def __init__(self, input_shape=(36, 1), n_units=128, **kwargs):
        self.input_shape = input_shape
        self.n_units = n_units
        self.optimizer = AdamW(learning_rate=kwargs.pop('learning_rate', 0.0001))
        self.num_layers = kwargs.pop('n_layers', 7)
        self.dropout = kwargs.pop('dropout', 0.3)
        self.model = self._build_model()

    def _build_model(self):
        """Builds the discriminator architecture."""
        data_input = Input(shape=self.input_shape)
        
        x = Flatten()(data_input)
        
        for _ in range(self.num_layers):
            x = Dense(self.n_units)(x)
            x = LeakyReLU(negative_slope=0.2)(x)
            x = Dropout(self.dropout)(x)

        output_prob = Dense(1)(x)
        
        model = Model(data_input, output_prob, name="discriminator")
        print("\n--- Discriminator Architecture ---")
        model.summary()
        return model

    def __call__(self, inputs):
        """Makes the class instance callable."""
        return self.model(inputs)
    
class GAN(tf.keras.Model):
    """
    The GAN model that combines the Generator and Discriminator.
    It takes a random noise vector and outputs a probability of the generated data being real.
    """
    def __init__(self, generator, discriminator, latent_dim=100, **kwargs):
        super().__init__(**kwargs)
        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = latent_dim
        self.gp_weight = kwargs.pop('gp_weight', 10.0)  # Gradient penalty weight

        self.generator_optimizer = None
        self.discriminator_optimizer = None

        self.generator_loss_metric = tf.keras.metrics.Mean(name='generator_loss')
        self.discriminator_loss_metric = tf.keras.metrics.Mean(name='discriminator_loss')

    @property
    def metrics(self):
        return [self.generator_loss_metric, self.discriminator_loss_metric]
    
    def compile(self, g_optimizer, d_optimizer):
        """
        Configures the model for training.
        We assign the optimizers passed here to the model.
        """
        super().compile()
        self.generator_optimizer = g_optimizer
        self.discriminator_optimizer = d_optimizer

    def gradient_penalty(self, batch_size, real_data, fake_data):
        """Calculates the gradient penalty."""
        alpha = tf.random.normal([batch_size, 1, 1], 0.0, 1.0)
        diff = fake_data - real_data
        interpolated = real_data + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            pred = self.discriminator(interpolated)

        grads = gp_tape.gradient(pred, [interpolated])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train_step(self, real_data):
        """Performs a single training step."""
        batch_size = tf.shape(real_data)[0]
        
        # Generate random noise

        for _ in range(5):
            noise = tf.random.normal([batch_size, self.latent_dim])

            with tf.GradientTape() as tape:
                fake_data = self.generator(noise)
                real_output = self.discriminator(real_data)
                fake_output = self.discriminator(fake_data)

                discriminator_loss = tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)
                gp = self.gradient_penalty(batch_size, real_data, fake_data)
                discriminator_loss += gp * self.gp_weight
            
            discriminator_gradients = tape.gradient(discriminator_loss, self.discriminator.model.trainable_variables)
            self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients, self.discriminator.model.trainable_variables))

        noise = tf.random.normal([batch_size, self.latent_dim])
        with tf.GradientTape() as tape:
            generated_data = self.generator(noise)
            gen_output = self.discriminator(generated_data)
            generator_loss = -tf.reduce_mean(gen_output)
        
        generator_gradients = tape.gradient(generator_loss, self.generator.model.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(generator_gradients, self.generator.model.trainable_variables))

        self.generator_loss_metric.update_state(generator_loss)
        self.discriminator_loss_metric.update_state(discriminator_loss) 

        return {
            'generator_loss': self.generator_loss_metric.result(),
            'discriminator_loss': self.discriminator_loss_metric.result()
        }



        
       
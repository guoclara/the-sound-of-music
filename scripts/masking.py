def mask(input, lmb=1):
  """
  input: tensor of shape (btch_size, x, y, channels)
  lmb: scaling factor for mask (the smaller lmb, the larger the mask)

  returns: Tensor masks of size (btch_size, x, y, channels)
  """
  assert input.shape[1] == input.shape[2], "Spectrograms not squares"

  btch_sz = input.shape[0]
  channels = input.shape[3]
  height_width = input.shape[1]
  
  np_input = input.numpy()
  np_input = np.swapaxes(np_input,1,3)
  np_input = np.swapaxes(np_input,2,3)
  
  #Referenced stack overflow for this function (https://stackoverflow.com/questions/30589211/numpy-argmax-over-multiple-axes-without-loop)
  def argmax_coord(A, N):
    s = A.shape
    new_shape = s[:-N] + (np.prod(s[-N:]),)
    max_idx = A.reshape(new_shape).argmax(-1)
    return np.unravel_index(max_idx, s[-N:])

  coordinates = argmax_coord(np_input, 2)
  
  #Referenced public implementation for this portion of code (https://github.com/andrehuang/InterpretableCNN)
  mu_x = np.reshape(np.array(coordinates[0]), (btch_sz, 1, 1, channels))
  mu_y = np.reshape(np.array(coordinates[1]), (btch_sz, 1, 1, channels))

  mu_x = mu_x/((height_width-1.0)/2.0) - 1.0
  mu_y = mu_y/((height_width-1.0)/2.0) - 1.0
  temp_x = np.reshape(np.linspace(-1, 1, height_width), (1, height_width, 1, 1))
  temp_y = np.reshape(np.linspace(-1, 1, height_width), (1, 1, height_width, 1))
  
  pos_temp_x = np.tile(temp_x, (btch_sz, 1, height_width, channels))
  pos_temp_y = np.tile(temp_y, (btch_sz, height_width, 1, channels))
  
  mu_x = np.tile(mu_x, (1, height_width, height_width, 1))
  mu_y = np.tile(mu_y, (1, height_width, height_width, 1))

  mask = np.absolute(pos_temp_x - mu_x)
  mask = np.add(mask, np.absolute(pos_temp_y - mu_y))
  mask = np.maximum(1 - np.multiply(mask, lmb), -1)

  return tf.convert_to_tensor(mask)


1. If you want to use any of the functionality here, you need Matlab or Octave. (Matlab recommended during training due to robustness and speed issues with Octave)
2. cd to this folder.
3. if you want to mess around with your own pics they need to be put in folder training/lets/
   [note: works only with *.jpg if you want to change it during learning faze (covered later) change init.m
   (most likely you will not need that, because training set is .jpg and it's quite big, but you could extend just file init.m it doesn't matter anywhere else in program)
	but at this stage you may want to change letDoIt.m ]
	after putting you pics into training/lets/ just run letDoIt.m
!!!!IMPORTANT: pics must be almost square ( YxY pixels) and face must be right in the middle because of the idiosyncrasies of this particular neural net
	(it could be done using Computer Vision in one or two days, but I don't have time right now because this project is due tomorrow)
4. if you want to muck around with training: run init.m -- loads pics from training/1 and training/2 (must be only women of only men), makes them 80x80, converts to greyscale, saves to 'X_Y.mat'
												 do_pca.m -- does pca on the output of init.m saves to X.mat, saves additional stuff needed to recover and project 80x80 pics into X_norm.mat (which is not needed for later work of program and serves visualisation purposes only, but needed on new input)
												 main.m -- trains NN outputs it's correctness on Training and Validation sets (it doesn't vaidation set so I didn't use Test set) and saves the Net into thetas.m
	consequitively. (you'll need only everything it saves(!) so don't delete anything)
	NOTE: you'll need at least 500 mb disk space with this configuration.
	if you want to change other code and don't have any ML experience (other than init.m and letsDoIt.m) you'd be better off not doing so,
	because some things are poorly documented at this stage.
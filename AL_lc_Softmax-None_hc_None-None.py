from src.active_learner import active_learner

al = active_learner(n_initial_samples = 2448,
                    n_epochs = [10],
                    low_certainty_threshold = 1000,
                    low_certainty_method = 'Softmax',
                    softmax_uncertainty_method ='entropy',
		    BS = 128,
		    al_until_full_train_set = True)

al.do_active_learning()

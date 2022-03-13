# # install.packages( "yardstick" )
# library( yardstick )

# k - fold Cross Validation & Grid Search for Time Series
cvgsts <- function( k, p ) { # k: total number of folds, p: train_1's portion of test_1 ( p = train_1 / test_1 )

  # solve system of equations ( 2 x 2 ) to find train_1 and test_1: train_1 + ( k * test_1 ) = nrow( wind_data ), test_1 = train_1 / p
  train_1 <- floor( ( p * nrow( wind_data ) ) / ( k + p ) ); test_1 <- floor( nrow( wind_data ) / ( p + k ) )

  # 1st fold's portion of nrow( wind_data ): cvgsts_start
  cvgsts_start <- ( ( train_1 + test_1 ) / nrow( wind_data ) )

  # window slide step calculation: win_step
  win_step <- ( test_1 / nrow( wind_data ) )

  # creating the dataframe to be used for grid search ( ML algorithm: SLR )
  hp_grid_df <- data.frame( "intercept" = c( TRUE, FALSE ), "smae_train" = vector( "numeric", 2 ), "smae_test" = vector( "numeric", 2 ) )

  # creating the dataframe to be used for huber loss delta hyperparameter grid search
  smae_grid_df <- data.frame( "delta" = c( 0.25, 0.50, 0.75, 1 ) )

  # creating the list, which will host the regressors trained with different settings of hyperparameters ( "intercept", "delta" )
  regressor_hp_delta_list <- list( "regressor_true_delta_list" = vector( "list", nrow( smae_grid_df ) ),
                                   "regressor_false_delta_list" = vector( "list", nrow( smae_grid_df ) ) )

  # grid search for hyperparameter: "delta" ( huber loss function )
  # smae_delta_best <- 0

  for ( d in ( 1 : nrow( smae_grid_df ) ) ) {

    smae_delta <- as.numeric( smae_grid_df[ d, 1 ] )
    smae_delta_best <- smae_delta

    # grid search for hyperparameter: "intercept" ( Simple Linear Regression )
    for ( g in ( 1 : nrow( hp_grid_df ) ) ) {

      smae_hp_train <- c()
      smae_hp_test <- c()
      hp <- data.frame( "intercept" = hp_grid_df[ g, 1 ] )

      # nested k - fold cross validation
      for ( i in seq( cvgsts_start, 1, win_step ) ) {

        fold <- wind_data[ ( 1 : floor( i * nrow( wind_data ) ) ), ]
        test <- fold[ ( ( ( nrow( fold ) - floor( ( test_1 / nrow( wind_data ) ) * nrow( wind_data ) ) ) + 1 ) : nrow( fold ) ), ]
        train <- fold[ ( 1 : ( nrow( fold ) - nrow( test ) ) ), ]

        # data preprocessing

        # data preprocessing: NAs' handling
        for ( mv in ( 1 : ( ncol( wind_data ) - 3 ) ) ) {

          train[ , mv ][ is.na( train[ , mv ] ) == TRUE ] <- mean( train[ , mv ], na.rm = TRUE )
          test[ , mv ][ is.na( test[ , mv ] ) == TRUE ] <- mean( train[ , mv ], na.rm = TRUE )

        }

        # data preprocessing: feature engineering
        for ( fs in ( 2 : ( ncol( wind_data ) - 3 ) ) ) {

          mean_train <- mean( train[ , fs ] )
          sd_train <- sd( train[ , fs ] )

          train[ , fs ] <- ( ( train[ , fs ] - mean_train ) / sd_train )
          test[ , fs ] <- ( ( test[ , fs ] - mean_train ) / sd_train ) ### here we use mean( train ) and not mean( test ), so that we avoid data leakage!

        }

        # training part: Machine Learning Algorithm ( SLR, MLR, ENR, PL, RFR, SVR, SARIMAX, ... ) & storing regressors into list: regressor_hp_delta_list
        # install.packages( "caret" )
        library( caret )

        if ( g == 1 ) {
          
          regressor_hp_delta_list[[ g ]][[ d ]] <- caret::train( weg ~ ws,
                                                                 data = train,
                                                                 method = "lm",
                                                                 tuneGrid = hp )

        } else if ( g == 2 ) {
          
          regressor_hp_delta_list[[ g ]][[ d ]] <- caret::train( weg ~ ws,
                                                                 data = train,
                                                                 method = "lm",
                                                                 tuneGrid = hp )
 
        }

        # testing / evaluation part: storing smae_train & smae_test ###
        # smae_train
        smae_hp_train <- append( smae_hp_train, huber_loss_vec( truth = as.numeric( train[ , 4 ] ),
                                                                estimate = as.numeric( unlist( regressor_hp_delta_list[[ g ]][[ d ]][[ 11 ]][ 5 ] ) ),
                                                                delta = smae_delta ) )
        
        # smae_test
        smae_hp_test <- append( smae_hp_test, huber_loss_vec( truth = as.numeric( test[ , 4 ] ),
                                                              estimate = as.numeric( predict( regressor_hp_delta_list[[ g ]][[ d ]], newdata = test ) ),
                                                              delta = smae_delta ) )

      }

      # filling the blanks in dataframe: "hp_grid_df" with mean( smae_train_hp ) and mean( smae_test_hp ) wrt: smae "delta" hyperparameter
      if ( mean( hp_grid_df[ , 3 ] ) == 0 ) {

        smae_delta_best <- smae_delta
        smae_delta_best_index <- d
        hp_grid_df[ g, 2 ] <- mean( smae_hp_train )
        hp_grid_df[ g, 3 ] <- mean( smae_hp_test )

      } else {
        
        if ( ( g == 2 ) & identical( hp_grid_df[ g, 2 ] == 0, hp_grid_df[ g, 2 ] == 0 ) ) {
          
          smae_delta_best <- smae_delta
          smae_delta_best_index <- d
          hp_grid_df[ g, 2 ] <- mean( smae_hp_train )
          hp_grid_df[ g, 3 ] <- mean( smae_hp_test )
        
        } else if ( ( g == 1 ) & ( mean( smae_hp_test ) < hp_grid_df[ g, 3 ] ) ) {
  
            smae_delta_best <- smae_delta
            smae_delta_best_index <- d
            hp_grid_df[ g, 2 ] <- mean( smae_hp_train )
            hp_grid_df[ g, 3 ] <- mean( smae_hp_test )
  
        } else if ( ( g == 2 ) & ( mean( smae_hp_test ) < hp_grid_df[ g, 3 ] ) ) {
  
            smae_delta_best <- smae_delta
            smae_delta_best_index <- d
            hp_grid_df[ g, 2 ] <- mean( smae_hp_train )
            hp_grid_df[ g, 3 ] <- mean( smae_hp_test )
          
        }
        
      }
      
    } # end of "intercept" hyperparameter tuning
  
  } # end of "delta" hyperparameter tuning

  # choosing the best "delta" and "intercept" hyperparameter settings and best regressor
  best_regressor <-  regressor_hp_delta_list[[ which( hp_grid_df[ , 3 ] == min( hp_grid_df[ , 3 ] ) ) ]][[ smae_delta_best_index ]]
  
  return( list( "regressor" = best_regressor, "smae" = hp_grid_df, "delta_best" = smae_delta_best ) )

}
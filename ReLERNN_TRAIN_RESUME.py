#!/usr/bin/env python
"""Resume training from a checkpoint - optimized for batch jobs"""

from ReLERNN.imports import *
from ReLERNN.helpers import *
from ReLERNN.sequenceBatchGenerator import *
from ReLERNN.networks import *


def runModels_resume(ModelFuncPointer,
            ModelName,
            TrainDir,
            TrainGenerator,
            ValidationGenerator,
            TestGenerator,
            resultsFile=None,
            numEpochs=10,
            epochSteps=100,
            validationSteps=1,
            network=None,
            nCPU=1,
            gpuID=0,
            initial_epoch=0):
    """Modified runModels that can resume from checkpoint"""
    os.environ["CUDA_VISIBLE_DEVICES"]=str(gpuID)
    
    from tensorflow.compat.v1 import ConfigProto
    from tensorflow.compat.v1 import Session
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    Session(config=config)
    
    if resultsFile == None:
        resultsFilename = os.path.basename(trainFile)[:-4] + ".p"
        resultsFile = os.path.join("./results/",resultsFilename)
    
    # Get a sample batch to determine input dimensions
    x, y = TrainGenerator.__getitem__(0)
    
    # Check if we're resuming from a checkpoint
    resume_from_checkpoint = initial_epoch > 0 and os.path.exists(network[1])
    
    if resume_from_checkpoint:
        print(f"\n=== RESUMING TRAINING FROM EPOCH {initial_epoch} ===\n")

        # Load model.json
        jsonFILE = open(network[0], "r")
        loadedModel = jsonFILE.read()
        jsonFILE.close()
        model = model_from_json(loadedModel)
        
        # Compile model
        model.compile(optimizer='Adam', loss='mse')

        # Summarize model
        model.summary()

        # Load weights
        print(f"Loading weights from {network[1]}")
        model.load_weights(network[1])
        print("Successfully loaded checkpoint!")
    else:
        print("\n=== STARTING NEW TRAINING ===\n")
        # Create new model
        model = ModelFuncPointer(x, y)
        # Save model architecture
        if network != None:
            model_json = model.to_json()
            with open(network[0], "w") as json_file:
                json_file.write(model_json)
            print(f"Saved initial model architecture to {network[0]}")
    
    # Callbacks
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        verbose=1,
        min_delta=0.01,
        patience=100)
    
    callbacks_list = [
        early_stop_callback,
        ModelCheckpoint(
            filepath=network[1],
            monitor='val_loss',
            save_best_only=True,
            verbose=1),
        TerminateOnNaN()
    ]
    
    # Train
    print(f"\nTraining from epoch {initial_epoch} to {numEpochs}")
    if nCPU > 1:
        history = model.fit(TrainGenerator,
            steps_per_epoch=epochSteps,
            epochs=numEpochs,
            initial_epoch=initial_epoch,
            validation_data=ValidationGenerator,
            callbacks=callbacks_list,
            use_multiprocessing=True,
            max_queue_size=nCPU,
            workers=nCPU)
    else:
        history = model.fit(TrainGenerator,
            steps_per_epoch=epochSteps,
            epochs=numEpochs,
            initial_epoch=initial_epoch,
            validation_data=ValidationGenerator,
            callbacks=callbacks_list,
            use_multiprocessing=False)
    
    # Check if early stopping was triggered
    early_stopped = early_stop_callback.stopped_epoch > 0
    actual_final_epoch = len(history.history['loss']) + initial_epoch
    
    if early_stopped:
        print(f"\nEARLY STOPPING triggered at epoch {early_stop_callback.stopped_epoch}")
        print(f"Training stopped before reaching max epochs ({numEpochs})")
    
    # Load best weights and make predictions
    print("\nLoading best weights for final evaluation...")
    model.load_weights(network[1])
    
    x, y = TestGenerator.__getitem__(0)
    predictions = model.predict(x)
    
    history.history['loss'] = np.array(history.history['loss'])
    history.history['val_loss'] = np.array(history.history['val_loss'])
    history.history['predictions'] = np.array(predictions)
    history.history['Y_test'] = np.array(y)
    history.history['name'] = ModelName
    history.history['early_stopped'] = early_stopped
    history.history['stopped_epoch'] = early_stop_callback.stopped_epoch if early_stopped else actual_final_epoch
    history.history['initial_epoch'] = initial_epoch
    history.history['final_epoch'] = actual_final_epoch
    history.history['requested_epochs'] = numEpochs - initial_epoch
    history.history['target_epoch'] = numEpochs
    
    print("Results written to:", resultsFile)
    pickle.dump(history.history, open(resultsFile, "wb"))
    
    return early_stopped, actual_final_epoch


def plotResults_with_early_stop(resultsFile, saveas):
    """modified plotResults with early stopping indicator"""
    import matplotlib.pyplot as plt

    plt.rc('xtick', labelsize=6)
    plt.rc('ytick', labelsize=6)
    plt.rc('axes', labelsize=6)
    
    # Load results
    results = pickle.load(open(resultsFile, "rb"))
    
    # Create figure (similar to original)
    fig = plt.figure(figsize=(12, 6))
    
    # Plot predictions vs actual
    ax1 = plt.subplot(1, 2, 1)
    plt.scatter(results['Y_test'], results['predictions'])
    plt.plot([min(results['Y_test']), max(results['Y_test'])], 
             [min(results['Y_test']), max(results['Y_test'])], 
             'r--')
    plt.xlabel('True recombination rate')
    plt.ylabel('Predicted recombination rate')
    plt.title('Test Set Predictions')
    
    # Plot training history
    ax2 = plt.subplot(1, 2, 2)
    epochs = range(len(results['loss']))
    plt.plot(epochs, results['loss'], label='Training Loss')
    plt.plot(epochs, results['val_loss'], label='Validation Loss')
    
    # Add vertical line if early stopping occurred
    if results.get('early_stopped', False):
        stopped_epoch = results.get('stopped_epoch', len(results['loss']))
        # Adjust to be relative to this training run
        initial_epoch = results.get('initial_epoch', 0)
        relative_stop = stopped_epoch - initial_epoch
        if 0 <= relative_stop < len(results['loss']):
            plt.axvline(x=relative_stop, ymin=0, 
                        ymax=max([max(results['loss']), max(results['val_loss'])]), 
                        color='red', linestyle='--', 
                        linewidth=2, label=f'Early Stop (epoch {stopped_epoch})')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(saveas)
    plt.close()
    
    print(f"Results plot saved to: {saveas}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--projectDir', dest='outDir', help='Directory for all project output', default=None)
    parser.add_argument('--nEpochs', dest='nEpochs', help='Maximum number of epochs to train', type=int, default=200)
    parser.add_argument('--nValSteps', dest='nValSteps', help='Number of validation steps', type=int, default=20)
    parser.add_argument('-t','--nCPU', dest='nCPU', help='Number of CPUs to use', type=int, default=1)
    parser.add_argument('-s','--seed', dest='seed', help='Random seed', type=int, default=None)
    parser.add_argument('--gpuID', dest='gpuID', help='Identifier specifying which GPU to use', type=int, default=0)
    parser.add_argument('--resume', dest='resume', help='Resume from checkpoint', action='store_true')
    args = parser.parse_args()
    
    ## Set seed
    if args.seed:
        os.environ['PYTHONHASHSEED'] = str(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
    
    ## Set number of cores
    nProc = args.nCPU
    
    ## Set up directory structure
    if not args.outDir:
        print("Warning: No project directory found, using current working directory.")
        projectDir = os.getcwd()
    else:
        projectDir = args.outDir
    
    trainDir = os.path.join(projectDir, "train")
    valiDir = os.path.join(projectDir, "vali")
    testDir = os.path.join(projectDir, "test")
    networkDir = os.path.join(projectDir, "networks")
    
    ## Define output files
    test_resultFile = os.path.join(networkDir, "testResults.p")
    modelSave = os.path.join(networkDir, "model.json")
    weightsSave = os.path.join(networkDir, "weights.h5")
    
    ## Determine initial epoch if resuming
    initial_epoch = 0
    
    if args.resume:
        # Check if checkpoint exists
        if not os.path.exists(weightsSave):
            print("\n ERROR: --resume specified but no checkpoint (weights.h5) found!")
            print(f"Expected checkpoint at: {weightsSave}")
            print("Cannot resume. Run without --resume to start fresh training.\n")
            sys.exit(1)
        
        # Check if testResults.p exists
        if not os.path.exists(test_resultFile):
            print("\n ERROR: --resume specified but testResults.p not found!")
            print(f"Expected file at: {test_resultFile}")
            print("This means the previous run did not complete successfully.")
            print("RECOMMENDATION: Increase runtime or reduce --nEpochs in previous run.\n")
            sys.exit(1)
        
        if not os.path.exists(modelSave):
            print("\n ERROR: --resume specified but model.json not found!")
            print(f"Expected file at: {modelSave}")
            print("This means the previous run did not complete successfully.")
            print("Start a fresh training or create a model file")
        
        print(f"\n*** Found checkpoint at {weightsSave} ***")
        
        # Load previous results
        try:
            prev_results = pickle.load(open(test_resultFile, "rb"))
            
            # Get final epoch from previous run
            if 'final_epoch' in prev_results:
                initial_epoch = prev_results['final_epoch']
            elif 'loss' in prev_results:
                initial_epoch = prev_results.get('initial_epoch', 0) + len(prev_results['loss'])
            else:
                print("ERROR: Cannot determine final epoch from testResults.p")
                sys.exit(1)
            
            # Check if previous run completed successfully
            epochs_completed = prev_results['final_epoch'] - prev_results['initial_epoch']
            epochs_requested = prev_results.get('requested_epochs', prev_results['final_epoch'] - prev_results['initial_epoch'])
            
            # Check for incomplete run (allow 5 epoch buffer)
            if not prev_results.get('early_stopped', False) and epochs_completed < (epochs_requested - 5):
                print(f"\n ERROR: Previous training run appears incomplete!")
                print(f"  Requested epochs: {epochs_requested}")
                print(f"  Completed epochs: {epochs_completed}")
                print(f"  Started from: {prev_results['initial_epoch']}")
                print(f"  Ended at: {prev_results['final_epoch']}")
                print(f"\nThis suggests the job timed out before completing.")
                print(f"RECOMMENDATION: Increase --time in SLURM job or reduce --nEpochs")
                print(f"Cannot safely resume. Please address the runtime issue first.\n")
                sys.exit(1)
            
            # Warn about early stopping
            if prev_results.get('early_stopped', False):
                print(f"\n WARNING: Previous training stopped early at epoch {prev_results.get('stopped_epoch', 'unknown')}")
                print("The model has likely converged. Additional training may not improve results.\n")
            
            print(f"*** Resuming from epoch {initial_epoch} ***")
            print(f"*** Previous training: epochs {prev_results['initial_epoch']} → {prev_results['final_epoch']} ***")
            print(f"*** Final training loss: {prev_results['loss'][-1]:.4f} ***")
            print(f"*** Final validation loss: {prev_results['val_loss'][-1]:.4f} ***\n")
            
        except Exception as e:
            print(f"\n ERROR: Could not read testResults.p: {e}")
            print("The file may be corrupted or incomplete.")
            sys.exit(1)
    
    ## Identify padding required
    maxSimS = 0
    winFILE = os.path.join(networkDir, "windowSizes.txt")
    with open(winFILE, "r") as fIN:
        for line in fIN:
            maxSimS = max([maxSimS, int(line.split()[5])])
    
    maxSegSites = 0
    for ds in [trainDir, valiDir, testDir]:
        DsInfoDir = pickle.load(open(os.path.join(ds, "info.p"), "rb"))
        segSitesInDs = max(DsInfoDir["segSites"])
        maxSegSites = max(maxSegSites, segSitesInDs)
    maxSegSites = max(maxSegSites, maxSimS)
    
    ## Set network parameters
    bds_train_params = {
        'treesDirectory': trainDir,
        'targetNormalization': "zscore",
        'batchSize': 64,
        'maxLen': maxSegSites,
        'frameWidth': 5,
        'shuffleInds': True,
        'sortInds': False,
        'center': False,
        'ancVal': -1,
        'padVal': 0,
        'derVal': 1,
        'realLinePos': True,
        'posPadVal': 0,
        'seqD': None,
        'seed': args.seed
    }
    
    ## Dump batch pars
    batchParsFILE = os.path.join(networkDir, "batchPars.p")
    with open(batchParsFILE, "wb") as fOUT:
        pickle.dump(bds_train_params, fOUT)
    
    bds_vali_params = copy.deepcopy(bds_train_params)
    bds_vali_params['treesDirectory'] = valiDir
    bds_vali_params['batchSize'] = 64
    
    bds_test_params = copy.deepcopy(bds_train_params)
    bds_test_params['treesDirectory'] = testDir
    DsInfoDir = pickle.load(open(os.path.join(testDir, "info.p"), "rb"))
    bds_test_params['batchSize'] = DsInfoDir["numReps"]
    bds_test_params['shuffleExamples'] = False
    
    ## Define sequence batch generators
    train_sequence = SequenceBatchGenerator(**bds_train_params)
    vali_sequence = SequenceBatchGenerator(**bds_vali_params)
    test_sequence = SequenceBatchGenerator(**bds_test_params)
    
    ## Train network
    early_stopped, actual_final_epoch = runModels_resume(
        ModelFuncPointer=GRU_TUNED84,
        ModelName="GRU_TUNED84",
        TrainDir=trainDir,
        TrainGenerator=train_sequence,
        ValidationGenerator=vali_sequence,
        TestGenerator=test_sequence,
        resultsFile=test_resultFile,
        network=[modelSave, weightsSave],
        numEpochs=args.nEpochs,
        validationSteps=args.nValSteps, 
        nCPU=nProc,
        gpuID=args.gpuID,
        initial_epoch=initial_epoch)
    
    ## Save plot with epoch number in filename
    test_resultFig = os.path.join(networkDir, f"testResults_final_epoch_{actual_final_epoch}.pdf")
    
    ## Plot results with early stopping indicator
    plotResults_with_early_stop(
        resultsFile=test_resultFile, 
        saveas=test_resultFig)
    
    ## Print summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Training range: Epoch {initial_epoch} → {actual_final_epoch}")
    print(f"Epochs completed this run: {actual_final_epoch - initial_epoch}")
    print(f"Early stopping triggered: {'YES' if early_stopped else 'NO'}")
    print(f"Results plot: {test_resultFig}")
    print("="*60 + "\n")
    
    print("\n***ReLERNN_TRAIN_RESUME.py FINISHED!***\n")


if __name__ == "__main__":
    main()

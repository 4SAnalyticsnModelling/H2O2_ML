using Flux, Flux.Zygote, MLJ, CSV, DataFrames, Statistics;
cd(@__DIR__);
df = DataFrame(CSV.File("../data/H2O2_ML_data.csv"));

# CSV.write("outputs/Table S1. Reported H2O2 data gathered from literature review.csv", df);

df = df[:, [4, 7, 8, 9, 10, 11, 12, 13, 14]];
show(first(df, 10), allcols = true)
# println(unique(df.substrate))
# println(unique(df.anode_electrode))
# println(unique(df.cathode_electrode))
# println(unique(df.cathode_operating_mode))
# println(unique(df.separator))
for cols in 3:7
	df[!, cols] .= MLJ.categorical(df[:, cols])
end;
hot = MLJ.OneHotEncoder(ordered_factor = true);
mach = MLJ.fit!(machine(hot, df));
df = MLJ.transform(mach, df);
show(first(df, 10), allcols = true)

# df[!, :log_transformed_H2O2_production_rate] .= log.(df.production_rate);
# df = df[:, Not(:production_rate)];
# CSV.write("outputs/Table S2. Processed features and target variables from the reported data in Table S1 used in deep learning.csv", df);

x = df[:, 1:26];
J, N = size(x);

# Deep neural network with 4 hidden layers (DeepNet4HL)
mutable struct nnet_mod_builder4
    n1 :: Int
    n2 :: Int
    n3 :: Int
    n4 :: Int
end;
function nnet_build(nn :: nnet_mod_builder4, n_in, n_out)
    return Flux.Chain(Dense(n_in, nn.n1, Flux.NNlib.relu, init = Flux.kaiming_normal),
                 Dense(nn.n1, nn.n2, Flux.NNlib.relu, init = Flux.kaiming_normal),
                 Dense(nn.n2, nn.n3, Flux.NNlib.relu, init = Flux.kaiming_normal),
                 Dense(nn.n3, nn.n4, Flux.NNlib.relu, init = Flux.kaiming_normal),
                 Dense(nn.n4, n_out, init = Flux.kaiming_normal))
end;

# Cross-validating model performance
for (h2o2param, h2o2_target) in zip(["h2o2_production_rate"], [27])
	y = df[:, h2o2_target];
	model_no = "architechture_4_3-fold_CV_";
	MLJulFluxFun.flux_mod_eval(nnet_build(nnet_mod_builder4(N * 16, N * 12, N * 8, N * 4), N, 1), x, y, "nnet_model_" * model_no * h2o2param * "_trained_model", MLJulFluxFun.KFold_(3, true, 10), MLJulFluxFun.KFold_(3, true, 1), nothing, 500, true, 5, 5, true, Flux.Losses.mse, Flux.Optimise.Adam(0.0001), 10);
end;

for (h2o2param, h2o2_target) in zip(["h2o2_production_rate"], [27])
	y = df[:, h2o2_target];
	model_no = "architechture_4_5-fold_CV_";
	MLJulFluxFun.flux_mod_eval(nnet_build(nnet_mod_builder4(N * 16, N * 12, N * 8, N * 4), N, 1), x, y, "nnet_model_" * model_no * h2o2param * "_trained_model", MLJulFluxFun.KFold_(5, true, 6), MLJulFluxFun.KFold_(3, true, 1), nothing, 500, true, 5, 5, true, Flux.Losses.mse, Flux.Optimise.Adam(0.0001), 10);
end;

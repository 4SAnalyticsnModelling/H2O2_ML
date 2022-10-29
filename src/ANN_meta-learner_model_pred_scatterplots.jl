using Flux, Flux.Zygote, MLJ, CSV, DataFrames, Statistics, BSON, CairoMakie;
cd(@__DIR__);
df = DataFrame(CSV.File("../data/H2O2_ML_data.csv"));
df = df[:, [4, 7, 8, 9, 10, 11, 12, 13, 14]];
show(first(df, 10), allcols = true)
for cols in 3:7
	df[!, cols] .= MLJ.categorical(df[:, cols])
end;
hot = MLJ.OneHotEncoder(ordered_factor = true);
mach = MLJ.fit!(machine(hot, df));
df = MLJ.transform(mach, df);
show(first(df, 10), allcols = true)
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
meta_model = nnet_build(nnet_mod_builder4(N * 16, N * 12, N * 8, N * 4), N, 1);
meta_mod_3_fold = BSON.load("nnet_model_architechture_4_3-fold_CV_h2o2_production_rate_trained_model/saved_ensembled_meta_model(s)/trained_model.bson")[:flux_model];
meta_mod_5_fold = BSON.load("nnet_model_architechture_4_5-fold_CV_h2o2_production_rate_trained_model/saved_ensembled_meta_model(s)/trained_model.bson")[:flux_model];

for (p, q, r) in zip(Flux.params(meta_model), Flux.params(meta_mod_3_fold), Flux.params(meta_mod_3_fold))
	p .= (q .+ r) ./ 2.0
end;

# Scatter Plots - Predicted vs Observed

# H2O2 production rate

f = Makie.Figure(resolution = (800, 800))
ax = Makie.Axis(f[1, 1], xlabel = "Reported H₂O₂ production rate (kg m⁻³ d⁻¹)", ylabel = "Predicted H₂O₂ production rate (kg m⁻³ d⁻¹)", xticklabelsize = 24, yticklabelsize = 24, xlabelsize = 26, ylabelsize = 26);
y_obs = df.production_rate;
y_pred = vec(meta_model(Matrix(Matrix(x)')));
Makie.scatter!(y_obs, y_pred, color = :green, strokewidth = 1);
ln1 = Makie.lines!([extrema(vcat(y_pred, y_obs))[1], extrema(vcat(y_pred, y_obs))[2]], [extrema(vcat(y_pred, y_obs))[1], extrema(vcat(y_pred, y_obs))[2]], color = :blue);
Makie.text!("R² = " * string(round(Statistics.cor(y_obs, y_pred)^2.0, digits = 3)) * "\nRMSE = " * string(round(sqrt(Flux.Losses.mse(y_pred, y_obs)), digits = 3)) * " kg m⁻³ d⁻¹", color = :blue, position = (10, 1), align = (:left, :bottom));
Makie.axislegend(ax, [ln1], ["1:1 fit line"], position = :lt, labelsize = 26);
f

Makie.save("outputs/Fig. 3. Final ensemble model performance scatterplots for H2O2 production rate.png", px_per_unit = 3.3, f);

%Data Grapher
headers = nslatency.Properties.VariableNames;

partial_timings = [nslatency.mobile_processing nslatency.transmission nslatency.server_processing];
end_to_end = nslatency.end_to_end;

partial_timings = partial_timings/1000;
end_to_end = end_to_end/1000;

sums = partial_timings*[1;1;1];

labels = categorical(nslatency.layer_name,nslatency.layer_name);

bar(labels,partial_timings, 'stacked');

legend({'mobile processing','transmission','mobile processing'}, 'Location','Best');
